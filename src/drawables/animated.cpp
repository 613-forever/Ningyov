// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/drawable.h>

#include <dialog_video_generator/math/pos_arith.h>

namespace dialog_video_generator { namespace drawable {

Animated::Animated(std::shared_ptr<Drawable> target) : target(std::move(target)) {}

Animated::~Animated() = default;

Frames Animated::duration() const {
  return std::min(leastAnimationDuration(), target->duration());
}

Movement::Movement(std::shared_ptr<Drawable> target, Frames duration) :
Animated(std::move(target)), dur(duration), frameOffset{} {}

Movement::~Movement() = default;

std::size_t Movement::nextFrame(Frames duration) {
  std::size_t leastChangedBuffer = target->nextFrame(duration);
  if (duration > dur) {
    return leastChangedBuffer;
  }
  Vec2i nextFrameOffset = calculateOffset(duration);
  if (nextFrameOffset == frameOffset) {
    return leastChangedBuffer;
  } else {
    return bufferCount();
  }
}

Frames Movement::leastAnimationDuration() const {
  return dur;
}

void Movement::nextScene(bool stop, Frames duration) {
  // TODO: take off the wrapper?
  target->nextScene(stop, duration);
}

void Movement::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  target->addTask(offset + frameOffset, alpha, tasks);
}

} }
