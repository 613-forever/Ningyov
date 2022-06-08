// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

#include <dialog_video_generator/config.h>

#include <unordered_map>
#include <common613/file_utils.h>
#include <dialog_video_generator/common.h>

namespace dialog_video_generator { namespace config {
std::uint16_t FRAMES_PER_SECOND = 30;
std::uint16_t WIDTH = 1920, HEIGHT = 1080;
std::uint16_t GPU_MAX_THREAD_PER_BLOCK = 1024, CPU_THREAD_NUM = 4;
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

inline auto readInto(std::string& configItem) {
  return [&configItem](const char* str, const char* sentinel) {
    configItem = std::string(str, sentinel);
  };
}

template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
inline auto readInto(T& configItem) {
  return [&configItem](const char* str, const char* sentinel) {
    configItem = checked_cast<T>(std::strtoull(str, nullptr, 10));
  };
}

void loadConfig() {
  using namespace common613::file;
  const File file = open("config.txt", "r", std::nothrow);
  if (file == nullptr) {
    return;
  }
  std::unordered_map<std::string, std::function<void(const char*, const char*)>> resolver;
  resolver["height"] = resolver["h"] = readInto(config::HEIGHT);
  resolver["width"] = resolver["w"] = readInto(config::WIDTH);
  // for now, ignore 23.976 (24/1001), 29.997 (30/1001), 59.994 (60/1001) or alike ones.
  resolver["frames_per_second"] = resolver["fps"] = readInto(config::FRAMES_PER_SECOND);
  resolver["gpu_max_thread_per_block"] = readInto(config::GPU_MAX_THREAD_PER_BLOCK);
  resolver["cpu_thread_num"] = readInto(config::CPU_THREAD_NUM);
  resolver["log_level"] = [](const char* str, const char* sentinel) {
    boost::log::trivial::from_string(str, sentinel - str, config::LOG_LEVEL);
  };

#ifndef DIALOGVIDEOGENERATOR_SKIP_CUDA_CONFIGS
  resolver["width_batches"] = readInto(config::WIDTH_BATCHES);
#endif

#ifndef DIALOGVIDEOGENERATOR_SKIP_STAND_CONFIGS
  resolver["stand_pose_string_format"] = readInto(config::STAND_POSE_STRING_FORMAT);
  resolver["stand_pose_dir_format"] = readInto(config::STAND_POSE_DIR_FORMAT);
  resolver["stand_pose_file_format"] = readInto(config::STAND_POSE_FILE_FORMAT);
  resolver["stand_expr_eye_format"] = readInto(config::STAND_EXPR_EYE_FORMAT);
  resolver["stand_expr_eye_count"] = readInto(config::STAND_EXPR_EYE_COUNT);
  resolver["stand_expr_mouth_format"] = readInto(config::STAND_EXPR_MOUTH_FORMAT);
  resolver["stand_expr_mouth_count"] = readInto(config::STAND_EXPR_MOUTH_COUNT);
#endif

#ifndef DIALOGVIDEOGENERATOR_SKIP_FONT_CONFIGS
  resolver["font_dir"] = readInto(config::FONT_DIRECTORY);
  resolver["fonts"] = [](const char* str, const char* sentinel) {
    config::FONT_NAMES_AND_SIZES.clear();
    auto* prev = str;
    const char* cur;
    do {
      cur = std::strchr(prev, ';');
      if (cur == nullptr) {
        cur = sentinel;
      }
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
        config::FONT_NAMES_AND_SIZES.emplace_back(font::FontInfo{std::string(prev, cur), 64, 64});
      }
      prev = cur + 1;
    } while(*cur != '\0');
  };
#endif

  constexpr int MAX = 64;
  char buffer[MAX];
  while (std::fgets(buffer, MAX, file.get())) {
    if (buffer[0] == '\n') {
      continue;
    }
    auto end = buffer + std::strlen(buffer);
    if (*(end - 1) == '\n') {
      *--end = '\0';
    }
    auto equalSign = std::strchr(buffer, '=');
    resolver[std::string(buffer, equalSign)](equalSign + 1, end);
  }
}

} }
