// SPDX-License-Identifier: MIT
// Copyright (c) 2021 613_forever

#include <dialog_video_generator/common.h>

#include <iostream>
#include <iomanip>
#include <boost/smart_ptr.hpp>
#include <boost/core/null_deleter.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sinks/async_frontend.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <common613/file_utils.h>

#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <common613/assert.h>
#endif

namespace dialog_video_generator {

namespace config {
std::uint16_t FRAMES_PER_SECOND = 30;
std::uint16_t WIDTH = 1920, HEIGHT = 1080;
std::uint16_t GPU_MAX_THREAD_PER_BLOCK = 1024, CPU_THREADS_NUM = 4;
boost::log::trivial::severity_level LOG_LEVEL = boost::log::trivial::info;
}

void loadConfig() {
  using namespace common613::file;
  const File file = open("config.txt", "r", std::nothrow);
  if (file == nullptr) {
    return;
  }
  std::map<std::string, std::function<void(const char*)>> resolver;
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

  constexpr int MAX = 64;
  char buffer[MAX];
  while (std::fgets(buffer, MAX, file.get())) {
    auto equalSign = std::strchr(buffer, '=');
    resolver[std::string(buffer, equalSign)](equalSign + 1);
  }
}

void initLog() {
  boost::log::core::get()->remove_all_sinks();
  boost::log::add_common_attributes();
  auto sink = boost::make_shared<boost::log::sinks::asynchronous_sink<boost::log::sinks::text_ostream_backend>>();
  boost::shared_ptr<std::ostream> pStream(&std::cerr, boost::null_deleter{});
  sink->locked_backend()->add_stream(pStream);
  sink->set_filter(boost::log::trivial::severity >= config::LOG_LEVEL);
  sink->set_formatter(
      boost::log::expressions::stream
          << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%H:%M:%S.%f")
          << "[" << std::left << std::setw(5) << boost::log::trivial::severity << "] "
          << boost::log::expressions::message
  );
  boost::log::core::get()->add_sink(sink);
}

void initStdOut() {
#ifdef _MSC_VER
  int res = _setmode(_fileno(stdout), _O_BINARY);
  BOOST_LOG_TRIVIAL(debug) << fmt::format("_setmode returns {:x}", res);
  COMMON613_REQUIRE(res != -1, "_setmode error, errno={}", errno);
#else
  std::freopen(nullptr, "wb", stdout);
#endif
}

void init() {
  loadConfig();

  initLog();

  initStdOut();
}

}
