// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_COMMON_H
#define DIALOGVIDEOGENERATOR_COMMON_H

#include <string>
#include <memory>
#include <cstdint>
#include <common613/assert.h>

namespace dialog_video_generator {

using namespace std::literals;

namespace config {
extern std::uint16_t FRAMES_PER_SECOND;
extern std::uint16_t WIDTH, HEIGHT;
extern std::uint16_t GPU_MAX_THREAD_PER_BLOCK, CPU_THREADS_NUM;
};

void init();

}

#endif //DIALOGVIDEOGENERATOR_COMMON_H
