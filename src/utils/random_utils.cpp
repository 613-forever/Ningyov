// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/math/random_utils.h>

namespace ningyov { namespace random {

std::mt19937_64 gen(std::random_device{}()); // NOLINT(cert-err58-cpp)

} }
