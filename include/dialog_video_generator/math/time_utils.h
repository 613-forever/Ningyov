// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TIME_UTILS_H
#define DIALOGVIDEOGENERATOR_TIME_UTILS_H

#include <chrono>
#include <common613/arith_utils.h>
#include <dialog_video_generator/common.h>

namespace dialog_video_generator { namespace time {

struct Frames {
  int count;
};

COMMON613_NODISCARD
inline Frames operator+(Frames lhs, Frames rhs) {
  return {lhs.count + rhs.count};
}

COMMON613_NODISCARD
inline Frames operator-(Frames lhs, Frames rhs) {
  return {lhs.count - rhs.count};
}

COMMON613_NODISCARD
inline bool operator==(const Frames& lhs, const Frames& rhs) {
  return lhs.count == rhs.count;
}

COMMON613_NODISCARD
inline bool operator!=(const Frames& lhs, const Frames& rhs) {
  return lhs.count != rhs.count;
}

COMMON613_NODISCARD
inline bool operator<(const Frames& lhs, const Frames& rhs) {
  return lhs.count < rhs.count;
}

COMMON613_NODISCARD
inline bool operator<=(const Frames& lhs, const Frames& rhs) {
  return lhs.count <= rhs.count;
}

COMMON613_NODISCARD
inline bool operator>(const Frames& lhs, const Frames& rhs) {
  return lhs.count > rhs.count;
}

COMMON613_NODISCARD
inline bool operator>=(const Frames& lhs, const Frames& rhs) {
  return lhs.count >= rhs.count;
}

}

using time::Frames;

inline Frames operator ""_fr(unsigned long long i) {
  return Frames{common613::checked_cast<int>(i)};
}
inline Frames operator ""_sec(unsigned long long i) {
  return Frames{common613::checked_cast<int>(i * config::FRAMES_PER_SECOND)};
}
inline Frames operator ""_sec(long double i) {
  return Frames{common613::checked_cast<int>(std::round(i * config::FRAMES_PER_SECOND))};
}
inline Frames operator ""_min(unsigned long long i) {
  return Frames{common613::checked_cast<int>(i * 60 * config::FRAMES_PER_SECOND)};
}
inline Frames operator ""_min(long double i) {
  return Frames{common613::checked_cast<int>(std::round(i * 60 * config::FRAMES_PER_SECOND))};
}

}

#endif //DIALOGVIDEOGENERATOR_TIME_UTILS_H
