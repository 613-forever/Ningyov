// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/engine.h>

#include <dialog_video_generator/image.h>
#include <iostream>

namespace dialog_video_generator { namespace engine {

Engine::StdoutStreaming::StdoutStreaming() {
  BOOST_LOG_TRIVIAL(info) << fmt::format("Frames will be dumped into stdout.");
}

Engine::StdoutStreaming::~StdoutStreaming() = default;

void Engine::StdoutStreaming::handleFrame(const Engine* engine, size_t index) {
  engine->lastLayerRGB->write(std::cout);
}

void Engine::StdoutStreaming::cleanup(const Engine* engine) {
  std::cout.flush();
}

} }
