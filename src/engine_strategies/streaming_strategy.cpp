// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/engine.h>

#include <thread>
#include <cuda_runtime.h>
#include <dialog_video_generator/image.h>

namespace dialog_video_generator { namespace engine {

Engine::StdoutStreaming::StdoutStreaming() {
  BOOST_LOG_TRIVIAL(info) << fmt::format("Frames will be dumped into stdout.");
}

Engine::StdoutStreaming::~StdoutStreaming() = default;

void Engine::StdoutStreaming::handleFrame(const Engine* engine, int index) {
  engine->buffers[engine->getBufferCount() - 1]->copyToMemory3([/*this*/](CudaMemory&& cudaMemory, std::size_t size){
    common613::Memory memory(size);
    cudaMemcpy(memory.data(), cudaMemory.get(), size, cudaMemcpyDeviceToHost);
    std::fwrite(memory.data(), 1, size, stdout);
  });
}

void Engine::StdoutStreaming::cleanup(const Engine* engine) {
  std::fflush(stdout);
}

} }
