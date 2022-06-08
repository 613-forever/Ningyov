// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#pragma once
#ifndef NINGYOV_COMMON_FILTERS_H
#define NINGYOV_COMMON_FILTERS_H

#include <ningyov/drawables/filters.h>

namespace ningyov { namespace abstraction {

class BlackAndWhiteFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  BlackAndWhiteFilter();
};

class NostalgicFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  NostalgicFilter();
};

class DuskFilter : public drawable::LinearFilter {
  static std::int8_t filterData[12];
public:
  DuskFilter();
};

} }

#endif //NINGYOV_COMMON_FILTERS_H
