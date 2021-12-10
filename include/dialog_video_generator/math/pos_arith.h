// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_POS_ARITH_H
#define DIALOGVIDEOGENERATOR_POS_ARITH_H

#include <common613/vector_arith_utils.h>
#include <common613/compat/cpp17.h>
#include <dialog_video_generator/math/pos_utils.h>

namespace dialog_video_generator {

template <class Val, class IntT>
COMMON613_NODISCARD
inline Val linear_interpolate(const Val& start, const Val& end, IntT current, IntT total) {
  return (end * current + start * (total - current)) / total;
}

template <bool A, class IntT, std::size_t N>
COMMON613_NODISCARD
inline common613::ArrNi<A, IntT, N> linear_interpolate(
    const common613::ArrNi<A, IntT, N>& start, const common613::ArrNi<A, IntT, N>& end, IntT current, IntT total) {
  return common613::internal::binaryHelper<A>(
      start, end, [=](IntT l, IntT r) { return linear_interpolate(l, r, current, total); },
      std::make_index_sequence<N>{}
  );
}

COMMON613_NODISCARD
inline Range makeRange(Vec2i lt, Size sz) {
  return Range{
      Pos2i::of(lt.x(), lt.y()),
      Pos2i::of(lt.x() + sz.w(), lt.y() + sz.h())
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
