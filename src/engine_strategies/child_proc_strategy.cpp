// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#ifdef MADOMAGI_DIALOG_ENABLE_SAVE_VIDEO_IPC_STRATEGY

#include <algorithm> // "process.hpp" will trigger "std::transform" not find without this on Ubuntu16.04/boost1.76.
#include <boost/process.hpp> // must be above engine on Windows, to avoid C1189: including "WinSock.h" repeatedly.
#include <dialog_video_generator/engine.h>

#include <common613/compat/file_system.h>
#include <dialog_video_generator/drawable.h>

namespace dialog_video_generator { namespace engine {

struct Engine::ChildProcVideo::State {
  boost::process::child childProcess;
  std::unique_ptr<boost::process::opstream> stream;
};

Engine::ChildProcVideo::ChildProcVideo(std::string cacheDir, std::string name)
    : targetDir(std::move(cacheDir)), name(std::move(name)), state(nullptr) {}

Engine::ChildProcVideo::~ChildProcVideo() {
  assert(state == nullptr);
}

void Engine::ChildProcVideo::init(const Engine* engine) {
  if (!common613::filesystem::exists(targetDir)) {
    BOOST_LOG_TRIVIAL(info) << "Creating directories: " << targetDir;
    common613::filesystem::create_directories(targetDir);
  }
  auto ffmpegExecutable = boost::process::search_path("ffmpeg");
  COMMON613_REQUIRE(!ffmpegExecutable.empty(), "Error: FFMpeg executable is not found.");
  BOOST_LOG_TRIVIAL(debug) << fmt::format("FFMpeg executable found: \"{}\".", ffmpegExecutable.string());
  auto stream = std::make_unique<boost::process::opstream>();
  try {
    state = new State { boost::process::child(
        ffmpegExecutable,
        "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", std::to_string(config::WIDTH) + "x"s + std::to_string(config::HEIGHT),
        "-r", std::to_string(config::FRAMES_PER_SECOND),
        "-i", "-", "-c:v" "libx264", "-pix_fmt", "yuv420p", "-maxrate", "6000k",
        targetDir + name,
        boost::process::std_in < *stream) };
  } catch (boost::process::process_error& e) {
    delete state;
    state = nullptr;
    BOOST_LOG_TRIVIAL(error) << fmt::format("Caught a Boost.Process error when initiating a child process: {}.", e.what());
    throw;
  }
  state->stream = std::move(stream);
}

void Engine::ChildProcVideo::handleFrame(const Engine* engine, int index) {
  assert(state != nullptr);
  engine->lastLayerRGB->write(*state->stream);
}

void Engine::ChildProcVideo::cleanup(const Engine* engine) {
  if (state != nullptr) {
    if (state->childProcess.running()) {
      state->stream->pipe().close();
      state->childProcess.wait();
    }
    delete state;
    state = nullptr;
  }
}

} }

#endif
