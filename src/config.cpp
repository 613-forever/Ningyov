// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/config.h>

#include <unordered_map>
#include <common613/file_utils.h>
#include <dialog_video_generator/common.h>

namespace dialog_video_generator { namespace config {
std::uint16_t FRAMES_PER_SECOND = 30;
std::uint16_t WIDTH = 1920, HEIGHT = 1080;
std::uint16_t GPU_MAX_THREAD_PER_BLOCK = 1024, CPU_THREADS_NUM = 4;
boost::log::trivial::severity_level LOG_LEVEL = boost::log::trivial::info; // no header for this.
} }

#ifndef DIALOGVIDEOGENERATOR_SKIP_CUDA_CONFIGS
#include <dialog_video_generator/cuda/cuda_utils.h>
#endif
#ifndef DIALOGVIDEOGENERATOR_SKIP_STAND_CONFIGS
#include <dialog_video_generator/drawables/stand.h>
#endif
#ifndef DIALOGVIDEOGENERATOR_SKIP_FONT_CONFIGS
#include <dialog_video_generator/text/text_render_utils.h>
#endif

namespace dialog_video_generator { namespace config {

void loadConfig() {
  using namespace common613::file;
  const File file = open("config.txt", "r", std::nothrow);
  if (file == nullptr) {
    return;
  }
  std::unordered_map<std::string, std::function<void(const char*)>> resolver;
  resolver["height"] = resolver["h"] = [](const char* str) {
    config::HEIGHT = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["width"] = resolver["w"] = [](const char* str) {
    config::WIDTH = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["frames_per_second"] = resolver["fps"] = [](const char* str) {
    // for now, ignore 23.976 (24/1001), 29.997 (30/1001), 59.994 (60/1001) or alike ones.
    config::FRAMES_PER_SECOND = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["gpu_max_thread_per_block"] = [](const char* str) {
    config::GPU_MAX_THREAD_PER_BLOCK = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["cpu_thread_num"] = [](const char* str) {
    config::CPU_THREADS_NUM = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["log_level"] = [](const char* str) {
    boost::log::trivial::from_string(str, std::strlen(str), config::LOG_LEVEL);
  };

#ifndef DIALOGVIDEOGENERATOR_SKIP_CUDA_CONFIGS
  resolver["width_batches"] = [](const char* str) {
    config::WIDTH_BATCHES = checked_cast<std::uint16_t>(std::strtoul(str, nullptr, 10));
  };
#endif

#ifndef DIALOGVIDEOGENERATOR_SKIP_STAND_CONFIGS
  resolver["stand_pose_string_format"] = [](const char* str) {
    config::STAND_POSE_STRING_FORMAT = str;
  };
  resolver["stand_pose_dir_format"] = [](const char* str) {
    config::STAND_POSE_DIR_FORMAT = str;
  };
  resolver["stand_pose_file_format"] = [](const char* str) {
    config::STAND_POSE_FILE_FORMAT = str;
  };
  resolver["stand_expr_eye_format"] = [](const char* str) {
    config::STAND_EXPR_EYE_FORMAT = str;
  };
  resolver["stand_expr_eye_count"] = [](const char* str) {
    config::STAND_EXPR_EYE_COUNT = checked_cast<std::uint8_t>(std::strtoul(str, nullptr, 10));
  };
  resolver["stand_expr_mouth_format"] = [](const char* str) {
    config::STAND_EXPR_MOUTH_FORMAT = str;
  };
  resolver["stand_expr_mouth_count"] = [](const char* str) {
    config::STAND_EXPR_MOUTH_COUNT = checked_cast<std::uint8_t>(std::strtoul(str, nullptr, 10));
  };
#endif

#ifndef DIALOGVIDEOGENERATOR_SKIP_FONT_CONFIGS
  resolver["font_dir"] = [](const char* str) {
    config::FONT_DIRECTORY = str;
  };
  resolver["fonts"] = [](const char* str) {
    config::FONT_NAMES_AND_SIZES.clear();
    auto* prev = str;
    const char* cur;
    do {
      cur = std::strpbrk(prev, ";\n");
      auto* it = std::strchr(prev, ',');
      if (it != nullptr && it < cur) {
        std::string name(prev,  it);
        char* it2;
        auto width = checked_cast<std::uint16_t>(std::strtoul(it + 1, &it2, 10));
        if (*it2 == ',') {
          auto height = checked_cast<std::uint16_t>(std::strtoul(it2 + 1, nullptr, 10));
          config::FONT_NAMES_AND_SIZES.emplace_back(font::FontInfo{name, width, height});
        } else {
          config::FONT_NAMES_AND_SIZES.emplace_back(font::FontInfo{name, width, width});
        }
      } else {
        config::FONT_NAMES_AND_SIZES.emplace_back(font::FontInfo{std::string(prev, cur), 48, 48});
      }
      prev = cur + 1;
    } while(*cur != '\n');
  };
#endif

  constexpr int MAX = 64;
  char buffer[MAX];
  while (std::fgets(buffer, MAX, file.get())) {
    auto equalSign = std::strchr(buffer, '=');
    resolver[std::string(buffer, equalSign)](equalSign + 1);
  }
}

} }
