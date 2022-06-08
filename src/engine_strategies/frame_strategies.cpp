// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/engine.h>

#include <common613/compat/file_system.h>
#include <ningyov/drawable.h>

using namespace common613::filesystem;

namespace ningyov { namespace engine {

Engine::SaveFrameByFrame::SaveFrameByFrame(std::string targetDir, std::string format)
    : targetDir(std::move(targetDir)), format(std::move(format)) {
  BOOST_LOG_TRIVIAL(info) << fmt::format("Frames will be saved in: \"{}\".", this->targetDir);
}

void Engine::SaveFrameByFrame::handleFrame(const Engine* engine, size_t index) {
  engine->lastLayerRGB->write(targetDir, fmt::format(format, index), engine->counter);
}

void Engine::SaveFrameByFrame::init(const Engine* engine) {
  if (!exists(targetDir)) {
    BOOST_LOG_TRIVIAL(info) << "Creating directories: " << targetDir;
    create_directories(targetDir);
  }
}

Engine::SaveIntermediateResults::SaveIntermediateResults(std::string targetDir, std::string format)
    : targetDir(std::move(targetDir)), format(std::move(format)) {
  BOOST_LOG_TRIVIAL(info) << fmt::format("Intermediate results will be saved in: \"{}\".", this->targetDir);
}

void Engine::SaveIntermediateResults::handleFrame(const Engine* engine, size_t index) {
  for (int layerIndex = 0; layerIndex < engine->getBufferCount(); ++layerIndex) {
    engine->buffers[layerIndex]->toRawImage3().write(targetDir, fmt::format(format, index, layerIndex), engine->counter);
  }
}

void Engine::SaveIntermediateResults::init(const Engine* engine) {
  if (!exists(targetDir)) {
    BOOST_LOG_TRIVIAL(info) << "Creating directories: " << targetDir;
    create_directories(targetDir);
  }
}

} }
