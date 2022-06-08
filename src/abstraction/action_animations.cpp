// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/abstraction/action_animations.h>

#include <ningyov/math/pos_arith.h>

using common613::checked_cast;

namespace ningyov { namespace abstraction {

Shouting::Shouting(std::shared_ptr<Drawable> target)
    : Movement(std::move(target), time::seconds(LENGTH_SECOND)) {}

Shouting::~Shouting() = default;

Vec2i Shouting::calculateOffset(Frames duration) const {
  const long double halfwayMax = (LENGTH_SECOND / 2) * (LENGTH_SECOND / 2)
      * config::FRAMES_PER_SECOND * config::FRAMES_PER_SECOND;
  int cur = duration.x() * (dur - duration).x();
  return Vec2i{0, checked_cast<Dim>(
      duration < dur ?
      cur * config::HEIGHT / -16.0 / halfwayMax :
      0
  )};
}

Murmuring::Murmuring(std::shared_ptr<Drawable> target)
    : Movement(std::move(target), time::seconds(LENGTH_SECOND)) {}

Murmuring::~Murmuring() = default;

Vec2i Murmuring::calculateOffset(Frames duration) const {
  if (duration <= dur) {
    return Vec2i::of(std::sin(duration.x() * std::acos(-1.) * 20. / dur.x()) * 8, 0);
  } else {
    return Vec2i::of(0, 0);
  }
}

}}
