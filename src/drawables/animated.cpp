// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/drawable.h>

#include <dialog_video_generator/math/pos_arith.h>

namespace dialog_video_generator { namespace drawable {

Animated::Animated(std::shared_ptr<Drawable> target) : target(std::move(target)) {}

Animated::~Animated() = default;

Frames Animated::duration() const {
  return std::max(leastAnimationDuration(), target->duration());
}

Movement::Movement(std::shared_ptr<Drawable> target, Frames duration)
    : Animated(std::move(target)), dur(duration), frameOffset{} {}

Movement::~Movement() = default;

std::size_t Movement::nextFrame(Frames timeInShot) {
  std::size_t leastChangedBuffer = target->nextFrame(timeInShot);
  if (timeInShot > dur) {
    return leastChangedBuffer;
  }
  Vec2i nextFrameOffset = calculateOffset(timeInShot);
  auto changedCount = nextFrameOffset == frameOffset ? leastChangedBuffer : bufferCount();
  frameOffset = nextFrameOffset;
  return changedCount;
}

Frames Movement::leastAnimationDuration() const {
  return dur;
}

std::shared_ptr<Drawable> Movement::nextShot(bool stop, Frames duration) {
  auto targetInNextShot = target->nextShot(stop, duration);
  if (duration > dur) {
    return targetInNextShot;
  } else {
    target = targetInNextShot;
    return shared_from_this();
  }
}

void Movement::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  target->addTask(offset + frameOffset, alpha, tasks);
}

AlphaChange::AlphaChange(std::shared_ptr<Drawable> target, Frames duration)
    : Animated(std::move(target)), dur(duration), frameAlpha{} {}

AlphaChange::~AlphaChange() = default;

std::size_t AlphaChange::nextFrame(Frames timeInShot) {
  std::size_t leastChangedBuffer = target->nextFrame(timeInShot);
  if (timeInShot > dur) {
    return leastChangedBuffer;
  }
  int nextFrameAlpha = calculateAlpha(timeInShot);
  auto changedCount = nextFrameAlpha == frameAlpha ? leastChangedBuffer : bufferCount();
  frameAlpha = nextFrameAlpha;
  return changedCount;
}

Frames AlphaChange::leastAnimationDuration() const {
  return dur;
}

std::shared_ptr<Drawable> AlphaChange::nextShot(bool stop, Frames duration) {
  auto targetInNextShot = target->nextShot(stop, duration);
  if (duration > dur) {
    return targetInNextShot;
  } else {
    target = targetInNextShot;
    return shared_from_this();
  }
}

void AlphaChange::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  target->addTask(offset, alpha * frameAlpha / 16, tasks);
}

} }
