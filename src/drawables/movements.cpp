// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/drawables/movements.h>

#include <dialog_video_generator/math/pos_arith.h>

namespace dialog_video_generator { namespace drawable {

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur), start(startOffset), end(endOffset) {}

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : SimpleMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

SimpleMovement::~SimpleMovement() = default;

Vec2i SimpleMovement::calculateOffset(Frames timeInScene) const {
  return Vec2i::of(
      std::round(linear_interpolate<double>(start.x(), end.x(), timeInScene.x(), dur.x())),
      std::round(linear_interpolate<double>(start.y(), end.y(), timeInScene.x(), dur.x()))
  );
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
  return Vec2i::of(
      std::round(linear_interpolate<double>(start.x(), end.x(), tmp * tmp * tmp, dur.x() * dur.x() * dur.x())),
      std::round(linear_interpolate<double>(start.y(), end.y(), tmp * tmp * tmp, dur.x() * dur.x() * dur.x()))
  );
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
    auto x = std::round(linear_interpolate(double(start.x()),
                                           double(end.x()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    auto y = std::round(linear_interpolate(double(start.y()),
                                           double(end.y()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    return Vec2i::of(x, y);
//    auto interpolatedExt = linear_interpolate(startExt, endExt, tmp * tmp * tmp / 2, dur.x() * dur.x() * dur.x());
  } else {
    auto tmp = 2 * dur.x() - doubleTime.x();
    auto x = std::round(linear_interpolate(double(start.x()),
                                           double(end.x()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    auto y = std::round(linear_interpolate(double(start.y()),
                                           double(end.y()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    return Vec2i::of(x, y);
//    auto interpolatedExt = linear_interpolate(startExt, endExt, (tmp * tmp * tmp) / 2, dur.x() * dur.x() * dur.x());
  }
}

}}
