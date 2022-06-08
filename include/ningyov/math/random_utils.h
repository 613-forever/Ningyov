// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file
/// @brief Header file for global random generator.

#pragma once
#ifndef NINGYOV_RANDOM_UTILS_H
#define NINGYOV_RANDOM_UTILS_H

#include <random>

namespace ningyov { namespace random {

/// @brief Global random generator.
extern std::mt19937_64 gen;

} }

#endif //NINGYOV_RANDOM_UTILS_H
