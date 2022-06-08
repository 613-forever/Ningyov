// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file pos_utils.h
/// @brief Utils about @ref Pos2i and @ref Vec2i.

#pragma once
#ifndef NINGYOV_POS_UTILS_H
#define NINGYOV_POS_UTILS_H

#include <common613/vector_definitions.h>

namespace ningyov {

using Byte = std::uint8_t;
using Dim = std::int16_t;
using UDim = std::uint16_t;

/// @brief 2D positions of integers.
/// @note Always mark positions using @ref Positions to avoid logic bugs.
using Pos2i = common613::Arr2i<false, Dim>;
COMMON613_CHECK_SIZE(Pos2i);
/// @brief 2D vectors of integers.
/// @note Always mark positions using @ref Positions to avoid logic bugs.
using Vec2i = common613::Arr2i<true, Dim>;
COMMON613_CHECK_SIZE(Vec2i);

/// @brief 2D ranges.
/// Used to describe rectangles.
/// @note Unused now. May be used to optimize rectangle repaint in the future.
struct Range {
  Pos2i leftTop, rightBottom;

  COMMON613_NODISCARD constexpr Dim l() const { return leftTop.x(); }
  COMMON613_NODISCARD constexpr Dim t() const { return leftTop.y(); }
  COMMON613_NODISCARD constexpr Dim r() const { return rightBottom.x(); }
  COMMON613_NODISCARD constexpr Dim b() const { return rightBottom.y(); }
  COMMON613_NODISCARD Dim& l() { return leftTop.x(); }
  COMMON613_NODISCARD Dim& t() { return leftTop.y(); }
  COMMON613_NODISCARD Dim& r() { return rightBottom.x(); }
  COMMON613_NODISCARD Dim& b() { return rightBottom.y(); }

  COMMON613_INJECT_SIZE_FIELD(2 * Pos2i::COMMON613_INJECTED_SIZE);
};
COMMON613_CHECK_BINARY_USABLE(Range);

/// @brief 2D sizes.
/// Used to describe size of rectangles.
struct Size : private common613::ArrNi<false, UDim, 2> {
  constexpr Size(const Size& arr) = default; // prevent resolution problem when copying
  constexpr Size(const ArrNi& arr) : ArrNi{arr} {} // NOLINT(google-explicit-constructor)

  COMMON613_NODISCARD constexpr UDim h() const { return arr[0]; }
  COMMON613_NODISCARD constexpr UDim w() const { return arr[1]; }
  COMMON613_NODISCARD UDim& h() { return arr[0]; }
  COMMON613_NODISCARD UDim& w() { return arr[1]; }
  COMMON613_NODISCARD constexpr std::uint32_t total() const { return std::uint32_t(h()) * w(); }

  COMMON613_INHERIT_SIZE_FIELD(ArrNi);

  template <class IntT, class IntT2>
  constexpr static Size of(IntT height, IntT2 width) {
    return Size{ArrNi::of(height, width)};
  }
};
COMMON613_CHECK_BINARY_USABLE(Size);

/// @brief RGBA structs.
/// Used to describe colors.
struct Color4b : private common613::ArrNi<true, Byte, 4> {
  constexpr Color4b(const Color4b& arr) = default; // prevent resolution problem when copying
  constexpr Color4b(const ArrNi& arr) : ArrNi{arr} {} // NOLINT(google-explicit-constructor)

  COMMON613_NODISCARD constexpr Byte r() const { return arr[3]; }
  COMMON613_NODISCARD constexpr Byte g() const { return arr[2]; }
  COMMON613_NODISCARD constexpr Byte b() const { return arr[1]; }
  COMMON613_NODISCARD constexpr Byte a() const { return arr[0]; }

  COMMON613_NODISCARD Byte& r() { return arr[3]; }
  COMMON613_NODISCARD Byte& g() { return arr[2]; }
  COMMON613_NODISCARD Byte& b() { return arr[1]; }
  COMMON613_NODISCARD Byte& a() { return arr[0]; }

  COMMON613_INHERIT_SIZE_FIELD(ArrNi);

  template <class IntT, class IntT2, class IntT3, class IntT4>
  constexpr static Color4b of(IntT r, IntT2 g, IntT3 b, IntT4 a) {
    return Color4b{ArrNi::of(a, b, g, r)};
  }
};
COMMON613_CHECK_BINARY_USABLE(Color4b);

}

#endif //NINGYOV_POS_UTILS_H
