// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file config.h
/// @brief common configs

#ifndef DIALOGVIDEOGENERATOR_CONFIG_H
#define DIALOGVIDEOGENERATOR_CONFIG_H

#include <cstdint>
#include <boost/log/trivial.hpp>

namespace dialog_video_generator { namespace config {
/// @{
/// @brief Video parameter.
extern std::uint16_t FRAMES_PER_SECOND;
extern std::uint16_t WIDTH, HEIGHT;
/// @}
/// @{
/// @brief Hardware parameter. Limits usage if necessary.
extern std::uint16_t GPU_MAX_THREAD_PER_BLOCK, CPU_THREADS_NUM;
/// @}
/// @brief Log level.
extern boost::log::trivial::severity_level LOG_LEVEL;

void loadConfig();
} }

#endif //DIALOGVIDEOGENERATOR_CONFIG_H
