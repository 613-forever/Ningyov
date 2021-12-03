// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/math/random_utils.h>

namespace dialog_video_generator {

std::mt19937_64 gen(std::random_device{}()); // NOLINT(cert-err58-cpp)

}
