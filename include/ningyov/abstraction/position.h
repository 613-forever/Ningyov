// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#ifndef NINGYOV_POSITION_H
#define NINGYOV_POSITION_H

#include <ningyov/common.h>
#include <ningyov/math/pos_utils.h>

namespace ningyov {
namespace position {

constexpr const int POSITION_BASE_LEFT_SHIFT = 6;
constexpr const int POSITION_RESOLUTION_LEFT_SHIFT = 2;
constexpr const int POSITION_MAX_BIAS_LEFT_SHIFT = POSITION_RESOLUTION_LEFT_SHIFT + 3;

enum Position : std::int8_t {
  KEEP = 0,
  NO_VALID_POSITION = -1,

  MIDDLE = 1 << POSITION_BASE_LEFT_SHIFT,
  MIDDLE_LEFT = MIDDLE - (1 << POSITION_RESOLUTION_LEFT_SHIFT),
  MIDDLE_RIGHT = MIDDLE + (1 << POSITION_RESOLUTION_LEFT_SHIFT),
  LEFT = MIDDLE - (2 << POSITION_RESOLUTION_LEFT_SHIFT),
  RIGHT = MIDDLE + (2 << POSITION_RESOLUTION_LEFT_SHIFT),
  FAR_LEFT = MIDDLE - (3 << POSITION_RESOLUTION_LEFT_SHIFT),
  FAR_RIGHT = MIDDLE + (3 << POSITION_RESOLUTION_LEFT_SHIFT),
  BORDER_LEFT = MIDDLE - (4 << POSITION_RESOLUTION_LEFT_SHIFT),
  BORDER_RIGHT = MIDDLE + (4 << POSITION_RESOLUTION_LEFT_SHIFT),
};

inline Vec2i enumToPosition(std::int8_t position) {
  return Vec2i::of((config::WIDTH * (position - BORDER_LEFT) + (1 << (POSITION_MAX_BIAS_LEFT_SHIFT - 1)))
                       >> POSITION_MAX_BIAS_LEFT_SHIFT, config::HEIGHT * 17 / 16);
}

}

using position::Position; // use Pos2i to specify non-semantic positions.
using position::enumToPosition; // use Pos2i to specify non-semantic positions.

}

#endif //NINGYOV_POSITION_H
