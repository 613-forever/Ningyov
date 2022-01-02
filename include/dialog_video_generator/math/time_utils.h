// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file time_utils.h
/// @brief Utils about time point and durations by the frame.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TIME_UTILS_H
#define DIALOGVIDEOGENERATOR_TIME_UTILS_H

#include <chrono>
#include <dialog_video_generator/common.h>
#include <common613/vector_definitions.h>

namespace dialog_video_generator { namespace time {

/// @brief Wrapper for frame count as time unit, to avoid logic bugs.
struct Frames : common613::ArrNi<true, std::size_t, 1> {
  constexpr Frames(const Frames& other) = default; // prevent resolution problem when copying
  constexpr Frames(const ArrNi& arr) : ArrNi{arr} {} // NOLINT(google-explicit-constructor)
  template <class NumT>
  constexpr static Frames of(NumT num) { return Frames{ArrNi::of(num)}; }
};

constexpr Frames frames(unsigned long long i) {
  return Frames::of(i);
}
inline Frames seconds(unsigned long long i) {
  return Frames::of(i * config::FRAMES_PER_SECOND);
}
inline Frames seconds(long double i) {
  return Frames::of(std::round(i * config::FRAMES_PER_SECOND));
}
inline Frames minutes(unsigned long long i) {
  return Frames::of(i * 60 * config::FRAMES_PER_SECOND);
}
inline Frames minutes(long double i) {
  return Frames::of(std::round(i * 60 * config::FRAMES_PER_SECOND));
}

}

using time::Frames;

constexpr Frames operator ""_fr(unsigned long long i) {
  return time::frames(i);
}
constexpr Frames operator ""_frames(unsigned long long i) {
  return time::frames(i);
}
inline Frames operator ""_sec(unsigned long long i) {
  return time::seconds(i);
}
inline Frames operator ""_sec(long double i) {
  return time::seconds(i);
}
inline Frames operator ""_min(unsigned long long i) {
  return time::minutes(i);
}
inline Frames operator ""_min(long double i) {
  return time::minutes(i);
}

}

#endif //DIALOGVIDEOGENERATOR_TIME_UTILS_H
