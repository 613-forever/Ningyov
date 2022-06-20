// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file engine.h
/// @brief Rendering engine for a single act.

#pragma once
#ifndef NINGYOV_ENGINE_H
#define NINGYOV_ENGINE_H

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <ningyov/math/time_utils.h>

namespace ningyov {

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

namespace abstraction {
class Director;
}

namespace engine {

class Engine;

/**
 * @brief Interface for frame handling strategies.
 */
class Strategy {
public:
  virtual ~Strategy() = 0;
  /// @brief Initializes the strategy.
  virtual void init(const Engine*) {}
  /// @brief Handles a frame.
  virtual void handleFrame(const Engine* engine, size_t index) {}
  /// @brief Cleans up the strategy.
  virtual void cleanup(const Engine*) {}
};

/**
 * @class Engine
 * @brief Rendering engine.
 *
 * Before using the engine, register a series of strategies into it.
 *
 * We can push layers into it.
 * The engine will render the layers into a sequence of frames, called a @b shot.
 * (In fact, it should not be called so, but I don't know how to name it better.)
 * For every frame, the strategies are called, which can be used to save or output the frames.
 *
 * Clear the states, reset the shot, and another shot can be re-rendered with the layers.
 * Of course, layers can be changed, modified or just left untouched and reused, to make another sequence.
 *
 * Strategies may be rewritten with static morphism to optimize further.
 * But our language level is C++14.
 * Without template parameter deduction, it will be a bother to define an `Engine` variable.
 */
class Engine {
public:
  using Strategies = std::vector<std::unique_ptr<Strategy>>;
  friend class ::ningyov::abstraction::Director;

  /**
   * @brief Constructs the engine.
   * @param strategies The strategies registered in the engine.
   * @param start The frame number to start with. [Now unused, many layers may need changing if it is used.]
   */
  explicit Engine(Strategies&& strategies, Frames start = 0_fr);
  ~Engine();

  /// @brief Returns a reference to the underlying layers.
  std::vector<std::shared_ptr<Drawable>>& getLayers() { return layers; }

  /// @brief Clears states to prepare for another shot.
  void nextShot(bool stop = true);

  /// @brief Renders the shot, i.e., generates frames and calls strategies.
  void renderShot() const;

private:
  void prepareMiddleResultBuffers(std::size_t size) const;
  void renderFirstFrame() const;
  void renderNonFirstFrame(Frames timeInShot) const;
  void renderTasks(size_t startBuffer, size_t bg, const std::vector<DrawTask>& tasks, size_t skippedTaskNumber) const;

public:
  /// @brief Set a shot to wait after all animation layers finishes, for an extra length of @p fr.
  void setWaitLength(Frames fr);
  /// @brief Shortcut to @c setWaitLength followed by @c renderShot .
  void setWaitAndRender(Frames fr) {
    setWaitLength(fr);
    renderShot();
  }
  /// @brief Set a shot to be of a total length of @p fr.
  void setTotalLength(Frames fr);
  /// @brief Shortcut to @c setTotalLength followed by @c renderShot .
  void setTotalAndRender(Frames fr) {
    setTotalLength(fr);
    renderShot();
  }

  /// @brief Returns the length when at least one animation layer is active.
  Frames getActiveLength() const;
  /// @brief Returns the total length of the shot.
  Frames getTotalLength() const;

  /// @brief Returns the count of the buffers prepared for the layers.
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
  /// @brief Strategy saving the frame.
  class SaveFrameByFrame : public Strategy {
  public:
    SaveFrameByFrame(std::string targetDir, std::string format);
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, size_t index) override;
  private:
    std::string targetDir, format;
  };
  /// @brief Strategy saving all buffers of the frame.
  class SaveIntermediateResults : public Strategy {
  public:
    SaveIntermediateResults(std::string cacheDir, std::string format);
    void init(const Engine* engine) override;
    void handleFrame(const Engine* engine, size_t index) override;
  private:
    std::string targetDir, format;
  };
#ifdef NINGYOV_ENABLE_SAVE_VIDEO_STRATEGY
  /// @brief Strategy saving the frame using FFMpeg.
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
#ifdef NINGYOV_ENABLE_SAVE_VIDEO_GPU_STRATEGY
  /// @brief Strategy saving the frame using VideoCodec.
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
  /// @brief Strategy streaming the frame into stdout.
  class StdoutStreaming : public Strategy {
  public:
    StdoutStreaming();
    ~StdoutStreaming() override;
    void handleFrame(const Engine* engine, size_t index) override;
    void cleanup(const Engine* engine) override;
  };
#ifdef NINGYOV_ENABLE_SAVE_VIDEO_IPC_STRATEGY
  /// @brief Strategy starting a FFMpeg subprocess when initialized, and piping the frame into it.
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

#endif //NINGYOV_ENGINE_H
