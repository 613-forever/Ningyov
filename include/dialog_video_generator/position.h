// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#ifndef DIALOGGENERATOR_POSITION_H
#define DIALOGGENERATOR_POSITION_H

#include <dialog_video_generator/common.h>
#include <dialog_video_generator/math/pos_utils.h>
#include <common613/arith_utils.h>

namespace dialog_video_generator {
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

inline Pos2i enumToPosition(Position position) {
  return Pos2i{
      common613::checked_cast<Dim>((config::WIDTH * (position - BORDER_LEFT) + (1 << (POSITION_MAX_BIAS_LEFT_SHIFT - 1))) >> POSITION_MAX_BIAS_LEFT_SHIFT),
      common613::checked_cast<Dim>(config::HEIGHT)
  };
}

}

using position::Position; // use Pos2i to specify non-semantic positions.

}

#endif //DIALOGGENERATOR_POSITION_H
