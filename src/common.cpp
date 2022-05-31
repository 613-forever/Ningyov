// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2022 613_forever

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
#include <dialog_video_generator/config.h>

#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <common613/assert.h>
#endif

namespace dialog_video_generator {

namespace config {
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
  config::loadConfig();

  initLog();

  initStdOut();
}

}
