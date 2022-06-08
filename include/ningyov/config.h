// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

/// @file config.h
/// @brief common configs

#ifndef NINGYOV_CONFIG_H
#define NINGYOV_CONFIG_H

#include <cstdint>
#include <boost/log/trivial.hpp>

namespace ningyov { namespace config {
/// @{
/// @brief Video parameter.
extern std::uint16_t FRAMES_PER_SECOND;
extern std::uint16_t WIDTH, HEIGHT;
/// @}
/// @{
/// @brief Hardware parameter. Limits usage if necessary.
extern std::uint16_t GPU_MAX_THREAD_PER_BLOCK, CPU_THREAD_NUM;
/// @}
/// @brief Log level.
extern boost::log::trivial::severity_level LOG_LEVEL;

void loadConfig();
} }

#endif //NINGYOV_CONFIG_H
