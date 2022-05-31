// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file common.h
/// @brief Utils about common configurations.

#pragma once
#ifndef DIALOGVIDEOGENERATOR_COMMON_H
#define DIALOGVIDEOGENERATOR_COMMON_H

#include <string>
#include <memory>
#include <cstdint>
#include <common613/assert.h>
#include <common613/checked_cast.h>

namespace dialog_video_generator {

using namespace std::literals;
using common613::checked_cast;

/// @brief Global initialization. Loads configures, setting up logging, and so on.
void init();

}

#endif //DIALOGVIDEOGENERATOR_COMMON_H
