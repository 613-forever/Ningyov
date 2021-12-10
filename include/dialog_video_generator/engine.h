// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_ENGINE_H
#define DIALOGVIDEOGENERATOR_ENGINE_H

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <dialog_video_generator/math/time_utils.h>

namespace dialog_video_generator {

namespace drawable {
class Drawable;
}
using drawable::Drawable;

using CudaMemory = std::shared_ptr<unsigned char>;

namespace image {
class RawImage3;
class RawImage;
}
using image::RawImage3;
using image::RawImage;

struct DrawTask;

namespace director {
class Director;
}

namespace engine {

class Engine;

class Strategy {
public:
  virtual ~Strategy() = 0;
  virtual void init(const Engine*) {}
  virtual void handleFrame(const Engine* engine, size_t index) {}
  virtual void cleanup(const Engine*) {}
};

class Engine {
public:
  using Strategies = std::vector<std::unique_ptr<Strategy>>;
  friend class ::dialog_video_generator::director::Director;

  explicit Engine(Strategies&& strategies, Frames start = 0_fr);
  ~Engine();

  std::vector<std::shared_ptr<Drawable>>& getLayers() { return layers; }

  void nextScene(bool stop = true);

  void renderScene() const;

private:
  void prepareMiddleResultBuffers(std::size_t size) const;
  void renderFirstFrame() const;
  void renderNonFirstFrame(Frames timeInScene) const;
  void renderTasks(size_t startBuffer, size_t bg, const std::vector<DrawTask>& tasks, size_t skippedTaskNumber) const;

public:
  void setWaitLength(Frames fr);
  void setTotalLength(Frames fr);
  Frames getActiveLength() const;
  Frames getTotalLength() const;
  std::size_t getBufferCount() const;

private:
  // an object of Engine is constant semantically when its contents do not change, so they are mutable.
  std::vector<std::shared_ptr<Drawable>> layers;
  // buffers to reuse
  mutable std::vector<std::shared_ptr<RawImage>> buffers;
  mutable std::shared_ptr<RawImage3> lastLayerRGB;
  // buffer count for every layer
  mutable std::vector<std::size_t> bufferIndices;
  mutable std::vector<std::size_t> bufferCount;
  // time
  Frames start;
  Frames wait;
  mutable Frames activeCache;
  // save
  Strategies strategies;
  mutable std::atomic_int counter;

public:
  class SaveFrameByFrame : public Strategy {
  public:
    SaveFrameByFrame(std::string targetDir, std::string format);
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, size_t index) override;
  private:
    std::string targetDir, format;
  };
  class SaveIntermediateResults : public Strategy {
  public:
    SaveIntermediateResults(std::string cacheDir, std::string format);
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, size_t index) override;
  private:
    std::string targetDir, format;
  };
#ifdef DIALOG_VIDEO_GENERATOR_ENABLE_SAVE_VIDEO_STRATEGY
  class SaveVideo: public Strategy {
  public:
    SaveVideo(std::string cacheDir, std::string name);
    ~SaveVideo() override;
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, int index) override;
    void cleanup(const Engine* engine) override;

    struct State;
  private:
    State* state;
    std::string targetDir, name;
  };
#endif
#ifdef DIALOG_VIDEO_GENERATOR_ENABLE_SAVE_VIDEO_GPU_STRATEGY
  class SaveVideoGPU : public Strategy {
  public:
    SaveVideoGPU(std::string cacheDir, std::string name);
    ~SaveVideoGPU() override;
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, int index) override;
    void cleanup(const Engine* engine) override;

    struct State;
  private:
    State* state;
    std::string targetDir, name;
  };
#endif
  class StdoutStreaming : public Strategy {
  public:
    StdoutStreaming();
    ~StdoutStreaming() override;
    void handleFrame(const Engine* engine, size_t index) override;
    void cleanup(const Engine* engine) override;
  };
#ifdef DIALOG_VIDEO_GENERATOR_ENABLE_SAVE_VIDEO_IPC_STRATEGY
  class ChildProcVideo : public Strategy {
  public:
    ChildProcVideo(std::string cacheDir, std::string name);
    ~ChildProcVideo() override;
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, size_t index) override;
    void cleanup(const Engine* engine) override;

    struct State;
  private:
    State* state;
    std::string targetDir, name;
  };
#endif
};

}

using engine::Engine;

}

#endif //DIALOGVIDEOGENERATOR_ENGINE_H
