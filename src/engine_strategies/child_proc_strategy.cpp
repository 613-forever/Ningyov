// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#ifdef MADOMAGI_DIALOG_ENABLE_SAVE_VIDEO_IPC_STRATEGY

#include <algorithm> // "process.hpp" will trigger "std::transform" not find without this on Ubuntu.
#include <boost/process.hpp> // must be above engine on Windows, to avoid C1189: including "WinSock.h" repeatedly.
#include <dialog_video_generator/engine.h>

#include <cuda_runtime.h>
#include <common613/memory.h>
#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace engine {

struct Engine::ChildProcVideo::State {
  boost::process::child childProcess;
  boost::process::opstream stream;
};

Engine::ChildProcVideo::ChildProcVideo(std::string cacheDir, std::string name)
    : targetDir(std::move(cacheDir)), name(std::move(name)), state(nullptr) {}

Engine::ChildProcVideo::~ChildProcVideo() {
  assert(state == nullptr);
}

void Engine::ChildProcVideo::init(const Engine* engine) {
  if (!boost::filesystem::exists(targetDir)) {
    BOOST_LOG_TRIVIAL(info) << "Creating directories: " << targetDir;
    boost::filesystem::create_directories(targetDir);
  }
  std::string command = fmt::format(
      "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size {}x{} -r {} -i - -c:v libx265 -pix_fmt yuv420p -maxrate 6000k {}",
      config::WIDTH, config::HEIGHT, config::FRAMES_PER_SECOND,
      targetDir + name
      );
  state = new State;
  state->childProcess = boost::process::child(command, boost::process::std_in < state->stream);
}

void Engine::ChildProcVideo::handleFrame(const Engine* engine, int index) {
  engine->buffers[engine->getBufferCount() - 1]->copyToMemory3([this](CudaMemory&& cudaMemory, std::size_t size){
    common613::Memory memory(size);
    cudaMemcpy(memory.data(), cudaMemory.get(), size, cudaMemcpyDeviceToHost);
    state->stream.write(reinterpret_cast<char*>(memory.data()), common613::checked_cast<long>(size));
  });
}

void Engine::ChildProcVideo::cleanup(const Engine* engine) {
  if (state != nullptr) {
    if (state->childProcess.running()) {
      state->stream.pipe().close();
      state->childProcess.wait();
    }
    delete state;
    state = nullptr;
  }
}

} }

#endif
