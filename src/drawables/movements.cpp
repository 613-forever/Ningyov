// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/drawables/movements.h>

#include <ningyov/math/pos_arith.h>

namespace ningyov { namespace drawable {

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur), start(startOffset), end(endOffset) {}

SimpleMovement::SimpleMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : SimpleMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

SimpleMovement::~SimpleMovement() = default;

Vec2i SimpleMovement::calculateOffset(Frames timeInShot) const {
  return Vec2i::of(
      std::round(linear_interpolate<double>(start.x(), end.x(), timeInShot.x(), dur.x())),
      std::round(linear_interpolate<double>(start.y(), end.y(), timeInShot.x(), dur.x()))
  );
}

CubicEaseInMovement::CubicEaseInMovement(std::shared_ptr<Drawable> target,
                                         Vec2i startOffset, Vec2i endOffset, Frames dur)
    : Movement(std::move(target), dur),
      start(startOffset), end(endOffset) {}

CubicEaseInMovement::CubicEaseInMovement(std::shared_ptr<Drawable> target, Vec2i endOffset, Frames dur)
    : CubicEaseInMovement(std::move(target), Vec2i{0, 0}, endOffset, dur) {}

CubicEaseInMovement::~CubicEaseInMovement() = default;

Vec2i CubicEaseInMovement::calculateOffset(Frames timeInShot) const {
  auto tmp = timeInShot.x();
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

Vec2i CubicEaseOutMovement::calculateOffset(Frames timeInShot) const {
  auto tmp = dur.x() - timeInShot.x();
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

Vec2i CubicEaseInOutMovement::calculateOffset(Frames timeInShot) const {
  auto doubleTime = timeInShot * 2;
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
    auto x = std::round(linear_interpolate(double(end.x()),
                                           double(start.x()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    auto y = std::round(linear_interpolate(double(end.y()),
                                           double(start.y()),
                                           tmp * tmp * tmp / 2,
                                           dur.x() * dur.x() * dur.x()));
    return Vec2i::of(x, y);
//    auto interpolatedExt = linear_interpolate(startExt, endExt, (tmp * tmp * tmp) / 2, dur.x() * dur.x() * dur.x());
  }
}

}}
