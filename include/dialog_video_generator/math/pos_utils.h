// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_POS_UTILS_H
#define DIALOGVIDEOGENERATOR_POS_UTILS_H

#include <common613/vector_definitions.h>

namespace dialog_video_generator {

using Byte = std::uint8_t;
using Dim = std::int16_t;
using UDim = std::uint16_t;

using Pos2i = common613::Arr2i<false, Dim>;
COMMON613_CHECK_SIZE(Pos2i);
using Vec2i = common613::Arr2i<true, Dim>;
COMMON613_CHECK_SIZE(Vec2i);

struct Range {
  Pos2i leftTop, rightBottom;

  COMMON613_NODISCARD Dim l() const { return leftTop.x(); }
  COMMON613_NODISCARD Dim t() const { return leftTop.y(); }
  COMMON613_NODISCARD Dim r() const { return rightBottom.x(); }
  COMMON613_NODISCARD Dim b() const { return rightBottom.y(); }
  COMMON613_NODISCARD Dim& l() { return leftTop.x(); }
  COMMON613_NODISCARD Dim& t() { return leftTop.y(); }
  COMMON613_NODISCARD Dim& r() { return rightBottom.x(); }
  COMMON613_NODISCARD Dim& b() { return rightBottom.y(); }

  COMMON613_INJECT_SIZE_FIELD(2 * Pos2i::COMMON613_INJECTED_SIZE);
};
COMMON613_CHECK_SIZE(Range);

struct Size : private common613::ArrNi<false, UDim, 2> {
  constexpr Size(const Size& arr) = default; // prevent resolution problem when copying
  constexpr Size(const ArrNi& arr) : ArrNi{arr} {} // NOLINT(google-explicit-constructor)

  COMMON613_NODISCARD UDim h() const { return arr[0]; }
  COMMON613_NODISCARD UDim w() const { return arr[1]; }
  COMMON613_NODISCARD UDim& h() { return arr[0]; }
  COMMON613_NODISCARD UDim& w() { return arr[1]; }
  COMMON613_NODISCARD std::uint32_t total() const { return std::uint32_t(h()) * w(); }

  COMMON613_INHERIT_SIZE_FIELD(ArrNi);

  template <class IntT, class IntT2>
  constexpr static Size of(IntT height, IntT2 width) {
    return Size{ArrNi::of(height, width)};
  }
};
COMMON613_CHECK_SIZE(Size);

struct Color4b : private common613::ArrNi<true, Byte, 4> {
  constexpr Color4b(const Color4b& arr) = default; // prevent resolution problem when copying
  constexpr Color4b(const ArrNi& arr) : ArrNi{arr} {} // NOLINT(google-explicit-constructor)

  COMMON613_NODISCARD Byte r() const { return arr[3]; }
  COMMON613_NODISCARD Byte g() const { return arr[2]; }
  COMMON613_NODISCARD Byte b() const { return arr[1]; }
  COMMON613_NODISCARD Byte a() const { return arr[0]; }

  COMMON613_NODISCARD Byte& r() { return arr[3]; }
  COMMON613_NODISCARD Byte& g() { return arr[2]; }
  COMMON613_NODISCARD Byte& b() { return arr[1]; }
  COMMON613_NODISCARD Byte& a() { return arr[0]; }

  COMMON613_INHERIT_SIZE_FIELD(ArrNi);

  template <class IntT, class IntT2, class IntT3, class IntT4>
  constexpr static Color4b of(IntT r, IntT2 g, IntT3 b, IntT4 a) {
    return Color4b{ArrNi::of(r, g, b, a)};
  }
};
COMMON613_CHECK_SIZE(Color4b);

}

#endif //DIALOGVIDEOGENERATOR_POS_UTILS_H
