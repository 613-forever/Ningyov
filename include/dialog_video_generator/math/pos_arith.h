// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_POS_ARITH_H
#define DIALOGVIDEOGENERATOR_POS_ARITH_H

#include <vector>
#include <common613/assert.h>
#include <common613/arith_utils.h>
#include <common613/compat/cpp17.h>
#include <dialog_video_generator/math/pos_utils.h>

namespace dialog_video_generator {

template <bool A, bool B>
COMMON613_NODISCARD
inline Arr2i<A && B> operator+(const Arr2i<A>& lhs, const Arr2i<B>& rhs) {
  static_assert(A || B, "Point + Point is not allowed.");
  return Vec2i{common613::checked_cast<Dim>(lhs.arr[0] + rhs.arr[0]), common613::checked_cast<Dim>(lhs.arr[1] + rhs.arr[1])};
}

template <bool A, bool B>
COMMON613_NODISCARD
inline Arr2i<A == B> operator-(const Arr2i<A>& lhs, const Arr2i<B>& rhs) {
  static_assert(!A || B, "Vector - Point is not allowed.");
  return Vec2i{common613::checked_cast<Dim>(lhs.arr[0] - rhs.arr[0]), common613::checked_cast<Dim>(lhs.arr[1] - rhs.arr[1])};
}

COMMON613_NODISCARD
inline Vec2i operator*(const Vec2i& lhs, Dim mul) {
  return Vec2i{common613::checked_cast<Dim>(lhs.arr[0] * mul), common613::checked_cast<Dim>(lhs.arr[1] * mul)};
}

COMMON613_NODISCARD
inline Vec2i operator/(const Vec2i& lhs, Dim mul) {
  // integer division
  return Vec2i{static_cast<Dim>(lhs.arr[0] / mul), static_cast<Dim>(lhs.arr[1] / mul)};
}

template <class Val>
COMMON613_NODISCARD
inline Val linear_interpolate(const Val& start, const Val& end, Dim current, Dim total) {
  return (end * current + start * (total - current)) / total;
}

template <>
COMMON613_NODISCARD
inline Pos2i linear_interpolate<Pos2i>(const Pos2i& start, const Pos2i& end, Dim current, Dim total) {
  return Pos2i{
      linear_interpolate(start.arr[0], end.arr[0], current, total),
      linear_interpolate(start.arr[1], end.arr[1], current, total),
  };
}

COMMON613_NODISCARD
inline Range makeRange(Vec2i lt, Size sz) {
  return Range{
      {lt.x(), lt.y()},
      {common613::checked_cast<Dim>(lt.x() + sz.w()), common613::checked_cast<Dim>(lt.y() + sz.h())}
  };
}

COMMON613_NODISCARD
inline bool intersect(const Range& lhs, const Range& rhs) {
  return !(lhs.r() <= rhs.l() || lhs.l() >= rhs.r() || lhs.t() >= rhs.b() || lhs.b() <= rhs.t());
}

COMMON613_NODISCARD
inline bool anyIntersect(const std::vector<Range>& rs, Range range) {
  return std::any_of(rs.begin(), rs.end(), [&range](const Range& r) {
    return intersect(r, range);
  });
}

}

#endif //DIALOGVIDEOGENERATOR_POS_ARITH_H
