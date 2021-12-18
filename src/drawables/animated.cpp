// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

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

std::size_t Movement::nextFrame(Frames timeInScene) {
  std::size_t leastChangedBuffer = target->nextFrame(timeInScene);
  if (timeInScene > dur) {
    return leastChangedBuffer;
  }
  Vec2i nextFrameOffset = calculateOffset(timeInScene);
  auto changedCount = nextFrameOffset == frameOffset ? leastChangedBuffer : bufferCount();
  frameOffset = nextFrameOffset;
  return changedCount;
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

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur), start(startOffset), end(endOffset) {}

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : SimpleMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

SimpleMovement::~SimpleMovement() = default;

Vec2i SimpleMovement::calculateOffset(Frames timeInScene) const {
  return linear_interpolate(start, end, timeInScene.x(), dur.x());
}

CubicEaseInMovement::CubicEaseInMovement(std::shared_ptr<Drawable> target,
                                         Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur),
      start(startOffset), end(endOffset) {}

CubicEaseInMovement::CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : CubicEaseInMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

CubicEaseInMovement::~CubicEaseInMovement() = default;

Vec2i CubicEaseInMovement::calculateOffset(Frames timeInScene) const {
  auto tmp = timeInScene.x();
  auto x = std::round(linear_interpolate(double(start.x()), double(end.x()), tmp * tmp * tmp, dur.x() * dur.x() * dur.x()));
  auto y = std::round(linear_interpolate(double(start.y()), double(end.y()), tmp * tmp * tmp, dur.x() * dur.x() * dur.x()));
  return Vec2i::of(x, y);
//  auto interpolatedExt = linear_interpolate(startExt, endExt, tmp * tmp * tmp, dur.x() * dur.x() * dur.x());
}

CubicEaseOutMovement::CubicEaseOutMovement(std::shared_ptr<Drawable> target,
                                           Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur),
      start(startOffset), end(endOffset) {}

CubicEaseOutMovement::CubicEaseOutMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : CubicEaseOutMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

CubicEaseOutMovement::~CubicEaseOutMovement() = default;

Vec2i CubicEaseOutMovement::calculateOffset(Frames timeInScene) const {
  auto tmp = dur.x() - timeInScene.x();
  auto x = linear_interpolate(double(end.x()), double(start.x()), tmp * tmp * tmp, dur.x() * dur.x() * dur.x());
  auto y = linear_interpolate(double(end.y()), double(start.y()), tmp * tmp * tmp, dur.x() * dur.x() * dur.x());
  return Vec2i::of(x, y);
//  auto interpolatedExt = linear_interpolate(endExt, startExt, tmp * tmp * tmp, dur.x() * dur.x() * dur.x()); // int64 overflow ...
}

CubicEaseInOutMovement::CubicEaseInOutMovement(std::shared_ptr<Drawable> target,
                                               Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur),
      start(startOffset), end(endOffset) {}

CubicEaseInOutMovement::CubicEaseInOutMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : CubicEaseInOutMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

CubicEaseInOutMovement::~CubicEaseInOutMovement() = default;

Vec2i CubicEaseInOutMovement::calculateOffset(Frames timeInScene) const {
  auto doubleTime = timeInScene * 2;
  if (doubleTime < dur) {
    auto tmp = doubleTime.x();
    auto x = std::round(linear_interpolate(double(start.x()), double(end.x()), tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x()));
    auto y = std::round(linear_interpolate(double(start.y()), double(end.y()), tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x()));
    return Vec2i::of(x, y);
//    auto interpolatedExt = linear_interpolate(startExt, endExt, tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x());
  } else {
    auto tmp = 2 * dur.x() - doubleTime.x();
    auto x = std::round(linear_interpolate(double(start.x()), double(end.x()), tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x()));
    auto y = std::round(linear_interpolate(double(start.y()), double(end.y()), tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x()));
    return Vec2i::of(x, y);
//    auto interpolatedExt = linear_interpolate(startExt, endExt, (tmp * tmp * tmp) / 2, dur.x() * dur.x() * dur.x());
  }
}

}}
