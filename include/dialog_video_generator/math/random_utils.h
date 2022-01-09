// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Header file for global random generator.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_RANDOM_UTILS_H
#define DIALOGVIDEOGENERATOR_RANDOM_UTILS_H

#include <random>

namespace dialog_video_generator { namespace random {

/// @brief Global random generator.
extern std::mt19937_64 gen;

} }

#endif //DIALOGVIDEOGENERATOR_RANDOM_UTILS_H
