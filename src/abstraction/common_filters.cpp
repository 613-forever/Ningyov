// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/abstraction/common_filters.h>

namespace dialog_video_generator { namespace abstraction {

NostalgicFilter::NostalgicFilter() : LinearFilter(filterData) {}

std::int8_t NostalgicFilter::filterData[12] = {
    25, 22, 17,
    49, 44, 34,
    12, 11, 8,
    0, 0, 0,
};

} }
