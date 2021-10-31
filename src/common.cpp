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
#include <common613/arith_utils.h>

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
boost::log::trivial::severity_level level = boost::log::trivial::info;
}

void loadConfig() {
  using namespace common613::file;
  const File file = open("config.txt", "r", std::nothrow);
  if (file == nullptr) {
    return;
  }
  std::uint16_t value;
  constexpr int MAX = 20;
  char buffer[MAX];
  int i = 0;
  while (std::fgets(buffer, MAX, file.get())) {
    value = common613::checked_cast<std::uint16_t>(std::strtoul(buffer, nullptr, 10));
    switch (i) {
    case 0:
      config::HEIGHT = value;
      break;
    case 1:
      config::WIDTH = value;
      break;
    case 2:
      config::FRAMES_PER_SECOND = value;
      break;
    case 3:
      config::GPU_MAX_THREAD_PER_BLOCK = value;
      break;
    case 4:
      config::CPU_THREADS_NUM = value;
      break;
    default:
      break;
    }
    i++;
  }
}

void initLog() {
  boost::log::core::get()->remove_all_sinks();
  boost::log::add_common_attributes();
  auto sink = boost::make_shared<boost::log::sinks::asynchronous_sink<boost::log::sinks::text_ostream_backend>>();
  boost::shared_ptr<std::ostream> pStream(&std::cerr, boost::null_deleter{});
  sink->locked_backend()->add_stream(pStream);
  sink->set_filter(boost::log::trivial::severity >= config::level);
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
  COMMON613_REQUIRE(res != -1, "setmode error, errno={}", errno);
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
