// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_POS_UTILS_H
#define DIALOGVIDEOGENERATOR_POS_UTILS_H

#include <array>
#include <common613/struct_size_check.h>

namespace dialog_video_generator {

using Byte = std::uint8_t;
using Dim = std::int16_t;
using UDim = std::uint16_t;

template <bool Vec>
struct Arr2i {
  constexpr static const bool isVec = Vec;
  constexpr static const size_t arraySize = 2;
  std::array<Dim, 2> arr;

  Dim x() const { return arr[0]; }
  Dim y() const { return arr[1]; }
  Dim& x() { return arr[0]; }
  Dim& y() { return arr[1]; }

  Arr2i<!Vec> fromOrigin() {
    return Arr2i<!Vec>{arr};
  }

  COMMON613_INJECT_SIZE_FIELD(4);
};

using Pos2i = Arr2i<false>;
COMMON613_CHECK_SIZE(Pos2i);
using Vec2i = Arr2i<true>;
COMMON613_CHECK_SIZE(Vec2i);

struct Range {
  Pos2i leftTop, rightBottom;

  Dim l() const { return leftTop.x(); }
  Dim t() const { return leftTop.y(); }
  Dim r() const { return rightBottom.x(); }
  Dim b() const { return rightBottom.y(); }
  Dim& l() { return leftTop.x(); }
  Dim& t() { return leftTop.y(); }
  Dim& r() { return rightBottom.x(); }
  Dim& b() { return rightBottom.y(); }

  COMMON613_INJECT_SIZE_FIELD(8);
};
COMMON613_CHECK_SIZE(Range);

struct Size {
  std::array<UDim, 2> arr;

  UDim h() const { return arr[0]; }
  UDim w() const { return arr[1]; }
  UDim& h() { return arr[0]; }
  UDim& w() { return arr[1]; }
  std::uint32_t total() const { return std::uint32_t(h()) * w(); }

  COMMON613_INJECT_SIZE_FIELD(4);
};
COMMON613_CHECK_SIZE(Size);

struct Color4b {
  Byte arr[4];

  Byte r() const { return arr[3]; }
  Byte g() const { return arr[2]; }
  Byte b() const { return arr[1]; }
  Byte a() const { return arr[0]; }

  Byte& r() { return arr[3]; }
  Byte& g() { return arr[2]; }
  Byte& b() { return arr[1]; }
  Byte& a() { return arr[0]; }

  COMMON613_INJECT_SIZE_FIELD(4);
};
COMMON613_CHECK_SIZE(Color4b);

}

#endif //DIALOGVIDEOGENERATOR_POS_UTILS_H
