// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <ningyov/drawables/filters.h>

namespace ningyov { namespace drawable {

Filter::~Filter() = default;

LinearFilter::LinearFilter(const std::int8_t (& data)[12]) {
  coefficients = cuda::copyFromCPUMemory(data, 12);
}

LinearFilter::~LinearFilter() = default;

void LinearFilter::addTask(Vec2i offset, unsigned int alpha, std::vector<DrawTask>& tasks) const {
  tasks.emplace_back(DrawTask{
      0, 0,
      config::HEIGHT, config::WIDTH,
      1,
      16,
      false, false,
      false, false, true, true,
      coefficients.get()
  });
}

} }
