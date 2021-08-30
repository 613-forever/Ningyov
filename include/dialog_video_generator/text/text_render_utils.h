// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#pragma once
#ifndef DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H
#define DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H

#include <string>

namespace dialog_video_generator { namespace font {

void init(const std::string& fontDir,
          const std::string& fontNameForChinese,
          const std::string& fontNameForJapanese);

} }

#endif //DIALOGVIDEOGENERATOR_TEXT_RENDER_UTILS_H
