// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/common_filters.h>

namespace dialog_video_generator { namespace abstraction {

NostalgicFilter::NostalgicFilter() : LinearFilter(filterData) {}

std::int8_t NostalgicFilter::filterData[12] = {
    25, 49, 12, 0,
    22, 44, 11, 0,
    17, 34, 8, 0,
};

} }
