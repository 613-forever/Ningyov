// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/engine.h>

#include <memory>
#include <thread>
#include <png.hpp>
#include <boost/filesystem.hpp>
#include <dialog_video_generator/drawable.h>
#include <common613/vector_arith_utils.h>

namespace dialog_video_generator { namespace engine {

Strategy::~Strategy() = default;

Engine::Engine(std::vector<std::unique_ptr<Strategy>>&& strategies, Frames s)
    : start(s), wait{0_fr}, activeCache{0_fr}, strategies(std::move(strategies)), counter(0), lastLayerRGB{} {
  for (auto& strategy: this->strategies) {
    strategy->init(this);
  }
}

Engine::~Engine() {
  for (auto& strategy: strategies) {
    strategy->cleanup(this);
    strategy = nullptr;
  }
  while (auto c = counter.load(std::memory_order_relaxed)) {
    BOOST_LOG_TRIVIAL(info) << fmt::format("Waiting for {} threads...", c);
    std::this_thread::sleep_for(1s);
  }
}

void Engine::nextShot(bool stop) {
  Frames length = getTotalLength();
  for (auto& layer: layers) {
    layer->nextShot(stop, length);
  }
  start = start + length;
  wait = 0_fr;
  activeCache = 0_fr;
}

void Engine::renderShot() const {
  // for frames render
  BOOST_LOG_TRIVIAL(info) << fmt::format("Rendering Frame {}.", start.x());
  renderFirstFrame();
  for (auto& strategy: strategies) {
    strategy->handleFrame(this, start.x());
  }

  for (auto time = 1_fr; time < getTotalLength(); time = time + 1_fr) {
    auto index = (start + time).x();
    BOOST_LOG_TRIVIAL(info) << fmt::format("Rendering Frame {}.", index);
    renderNonFirstFrame(time);
    COMMON613_REQUIRE(index > 0 && index < 1'0000'0000, "Index is invalid: {}", index);
    for (auto& strategy: strategies) {
      strategy->handleFrame(this, index);
    }
  }
  BOOST_LOG_TRIVIAL(trace) << "Finished rendering for one shot.";
}

void Engine::prepareMiddleResultBuffers(size_t size) const {
  if (buffers.size() < size) {
    auto oldSize = buffers.size();
    BOOST_LOG_TRIVIAL(debug) << fmt::format("Resizing buffer to {} images from {}.", size, oldSize);
    buffers.resize(size, nullptr);

    for (size_t i = oldSize; i < size; ++i) {
      if (buffers[i] == nullptr) {
        buffers[i] = RawImage::allocate(config::HEIGHT, config::WIDTH);
        BOOST_LOG_TRIVIAL(debug) << fmt::format(
              "Allocating GPU memory for middle results from Layer {} at {}x{}@{:p}",
              i, buffers[i]->size.h(), buffers[i]->size.w(),
              buffers[i]->memory.get());
      }
    }
  }
}

void Engine::renderFirstFrame() const {
  if (!lastLayerRGB) {
    lastLayerRGB = RawImage3::allocate(config::HEIGHT, config::WIDTH);
  }
  std::vector<DrawTask> tasks;
  bufferCount.resize(layers.size());
  bufferIndices.resize(layers.size());
  if (!layers.empty()) {
    bufferCount[0] = layers[0]->bufferCount();
    bufferIndices[0] = bufferCount[0] + 1;
    layers[0]->nextFrame(0_fr);
    layers[0]->addTask(Vec2i{0, 0}, 16, tasks);
    for (int i = 1; i < layers.size(); ++i) {
      bufferCount[i] = layers[i]->bufferCount();
      bufferIndices[i] = bufferIndices[i - 1] + bufferCount[i];
      layers[i]->nextFrame(0_fr);
      layers[i]->addTask(Vec2i{0, 0}, 16, tasks);
    }
    prepareMiddleResultBuffers(bufferIndices.back());
    for (std::size_t i = 0, buf = 1; i < layers.size(); ++i) {
      BOOST_LOG_TRIVIAL(trace) << (bufferCount[i] == 1 ? fmt::format("Layer {} will write 1 buffer [{}].", i, buf) :
                                   fmt::format("Layer {} will write {} buffers [{},{}).",
                                               i, bufferCount[i], buf, bufferIndices[i]));
      buf = bufferIndices[i];
    }
    renderTasks(1, 0, tasks, 0);
  }
}

void Engine::renderNonFirstFrame(Frames timeInShot) const {
  if (!layers.empty()) {
    std::vector<DrawTask> tasks;
    bool staticFlag = true;
    std::size_t skippedTask, layerDynamic, firstDynamicBuffer;
    for (auto i = 0; i < layers.size(); ++i) {
      if (staticFlag) {
        layerDynamic = layers[i]->nextFrame(timeInShot);
        if (layerDynamic == 0) {
          continue;
        } else {
          staticFlag = false;
          firstDynamicBuffer = bufferIndices[i] - layerDynamic;
          skippedTask = bufferCount[i] - layerDynamic;
        }
      } else {
        layers[i]->nextFrame(timeInShot);
      }
      layers[i]->addTask(Vec2i{0, 0}, 16, tasks);
    }
    if (tasks.empty()) return;
    if (!staticFlag) {
      renderTasks(firstDynamicBuffer, firstDynamicBuffer - 1, tasks, skippedTask);
    }
  }
}

void Engine::renderTasks(size_t startBuffer, size_t bg,
                         const std::vector<DrawTask>& tasks, size_t skippedTaskNumber) const {
  BOOST_LOG_TRIVIAL(trace) << fmt::format(
        "Rendering {} tasks (first {} skipped), writing from Buffer {} and use {} as background.",
        tasks.size(), skippedTaskNumber, startBuffer, bg);
  for (auto& task: tasks) {
    BOOST_LOG_TRIVIAL(trace) << fmt::format("Task ({},{}),{}x{},{}x, skipped:{}",
                                            task.x0, task.y0, task.ht, task.wt, task.mul, task.skip);
  }
  auto tasksSize = tasks.size() - skippedTaskNumber;
  auto tasksSizeInBytes = tasksSize * sizeof(DrawTask);
  auto cudaTasks = cuda::copyFromCPUMemory(tasks.data() + skippedTaskNumber, tasksSizeInBytes);

  size_t dstCount = getBufferCount() - startBuffer;
  auto bufferRequired = tasksSize;
  // std::count_if(tasks.begin() + common613::checked_cast<std::ptrdiff_t>(skippedTaskNumber),
  //              tasks.end(), [](const DrawTask& task) { return !task.skip; });
  COMMON613_REQUIRE(bufferRequired == dstCount,
                    "Mismatching numbers of buffers {} (required by tasks) and {} (provided to rewrite).",
                    bufferRequired, dstCount);
  std::vector<unsigned char*> destinations;
  destinations.reserve(dstCount + 1);
  destinations.push_back(buffers[bg]->memory.get());
  std::transform(buffers.begin() + common613::checked_cast<std::ptrdiff_t>(startBuffer),
                 buffers.begin() + common613::checked_cast<std::ptrdiff_t>(getBufferCount()),
                 std::back_inserter(destinations),
                 [](const std::shared_ptr<RawImage>& ptr) { return ptr->memory.get(); });
  assert(destinations.size() == dstCount + 1);
  auto destinationsOnGPU = cuda::copyFromCPUMemory(destinations.data(), destinations.size() * sizeof(unsigned char*));
  cuda::renderTasks(reinterpret_cast<unsigned char**>(destinationsOnGPU.get()),
                    reinterpret_cast<DrawTask*>(cudaTasks.get()), tasksSize,
                    lastLayerRGB ? (lastLayerRGB->invalidateLazyCPUMemory(), lastLayerRGB->memory.get()) : nullptr);
}

void Engine::setWaitLength(Frames fr) {
  wait = fr;
}

void Engine::setTotalLength(Frames fr) {
  wait = fr - getActiveLength();
}

Frames Engine::getActiveLength() const {
  if (activeCache.x() == 0) {
    Frames longest{1_fr}; // at least 1 frame
    for (const auto& layer: layers) {
      auto duration = layer->duration();
      if (duration > longest) {
        longest = duration;
      }
    }
    activeCache = longest;
  }
  return activeCache;
}

Frames Engine::getTotalLength() const {
  return wait + getActiveLength();
}

std::size_t Engine::getBufferCount() const {
  return bufferIndices.empty() ? 1 : bufferIndices.back();
}

} }
