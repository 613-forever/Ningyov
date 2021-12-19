// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/action_animation.h>

#include <dialog_video_generator/math/pos_arith.h>

using common613::checked_cast;

namespace dialog_video_generator { namespace abstraction {

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
  auto phase = dur.x() / 6;
  if (duration <= dur / 6) {
    return linear_interpolate(Vec2i::of(0, 0), Vec2i::of(config::HEIGHT / 18, config::HEIGHT / 24),
                              duration.x(), phase);
  } else if (duration <= dur / 3) {
    return linear_interpolate(Vec2i::of(config::HEIGHT / 18, config::HEIGHT / 24), Vec2i::of(-config::HEIGHT / 18, config::HEIGHT / 12),
                              duration.x() - phase, phase);
  } else if (duration <= dur / 2) {
    return linear_interpolate(Vec2i::of(-config::HEIGHT / 18, config::HEIGHT / 12), Vec2i::of(-config::HEIGHT / 18, config::HEIGHT / 8),
                              duration.x() - dur.x() / 3, phase);
  } else if (duration <= dur) {
    return Vec2i::of(0, config::HEIGHT * (duration.x() * 2 - dur.x()) / dur.x() / 8);
  } else {
    return Vec2i::of(0, 0);
  }
}

}}
