#ifndef __LOGGING_UTILS_H__
#define __LOGGING_UTILS_H__

/* set non-debug mode */
#ifndef NDEBUG
    #define NDEBUG 
#endif

/** set debug mode */
// #ifdef NDEBUG
//     #undef NDEBUG 
// #endif

#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#include <chrono>
#include <memory>
#include <iostream>
#include <cassert>
#include <string.h>

#define __LOG_USE_BASE_FILENAME__

#ifdef __LOG_USE_BASE_FILENAME__
    #define __LOG_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#else
    #define __LOG_FILENAME__  __FILE__
#endif

//** log with fprintf *//
#define LOGPF(format, ...) fprintf(stderr ,"[%s:%d] " format "\n", __LOG_FILENAME__, __LINE__, ##__VA_ARGS__)

/**
 * 1: ros1
 * 2: ros2
 * 3: spdlog
 * 4: se_osal_log
 * others: fprintf
 */

#define __LOGGING_BACKEND__  0

#if __LOGGING_BACKEND__ == 4
    #include <se_sal_types.h>
    #include <se_osal_log.h>

    #define RLOGD(...) SE_LOG_Debug(__VA_ARGS__)
    #define RLOGI(...) SE_LOG_Info(__VA_ARGS__)
    #define RLOGW(...) SE_LOG_Warning(__VA_ARGS__)
    #define RLOGE(...) SE_LOG_Error(__VA_ARGS__)
    #define RLOGF(...) SE_LOG_Error(__VA_ARGS__)

/** ros2 does built-in spdlog, just find_package(spdlog REQUIRED) and link to spdlog */
#elif __LOGGING_BACKEND__ == 3
    #include <spdlog/spdlog.h>
    #include <spdlog/fmt/bundled/printf.h>
    #include "spdlog/sinks/basic_file_sink.h"
    #include "spdlog/sinks/stdout_sinks.h"
    #include "spdlog/sinks/stdout_color_sinks.h"
    #include "spdlog/sinks/rotating_file_sink.h"

    template <class loggerPtr, class... Args>
    void loglineprintf(loggerPtr logger, spdlog::level::level_enum level, spdlog::source_loc loc, const char* fmt, const Args&... args) noexcept
    {
        if (logger && logger->should_log(level))
        {
            logger->log(loc, level, "{}", fmt::sprintf(fmt, args...));
        }
    }

    #define SPDLOG_LOGGER_PRINTF(logger, level, ...) \
        loglineprintf(logger, level, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, __VA_ARGS__)

    //specific log implementation macros

    #define RLOGD(...) SPDLOG_LOGGER_PRINTF(spdlog::default_logger(),spdlog::level::debug,__VA_ARGS__)
    #define RLOGI(...) SPDLOG_LOGGER_PRINTF(spdlog::default_logger(),spdlog::level::info,__VA_ARGS__)
    #define RLOGW(...) SPDLOG_LOGGER_PRINTF(spdlog::default_logger(),spdlog::level::warn,__VA_ARGS__)
    #define RLOGE(...) SPDLOG_LOGGER_PRINTF(spdlog::default_logger(),spdlog::level::err,__VA_ARGS__)
    #define RLOGF(...) SPDLOG_LOGGER_PRINTF(spdlog::default_logger(),spdlog::level::critical,__VA_ARGS__)

    static inline void __setup_spdlog__(const char* log_file, spdlog::level::level_enum log_level)
    {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
        /** auto-rotate log files, 20MB each, 5 in total, create new log file when process restart */
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file, 1024*1024*20, 5, true));
        auto combined_logger = std::make_shared<spdlog::logger>("combined_logger", begin(sinks), end(sinks));

        spdlog::register_logger(combined_logger);
        spdlog::set_default_logger(combined_logger);
        spdlog::flush_every(std::chrono::seconds(3));
        spdlog::set_level(log_level);
        /** [%g:%#] for full file path */
        spdlog::set_pattern("[%L][%t][%Y-%m-%d %H:%M:%S.%f][%s:%#] %v");
    }

    /** call this before progress exit */
    static inline void __shutdown_spdlog__()
    {
        spdlog::shutdown();
    }

    static inline void __run_spdlog_logger_test__()
    {
        __setup_spdlog__("/tmp/atlantis/spdlog_test.log", spdlog::level::info);
        float magic = 12.3f;
        RLOGD("this is debug: %.2f", magic);
        RLOGI("this is info: %.2f", magic);
        RLOGW("this is warn: %.2f", magic);
        RLOGE("this is error: %.2f", magic);
        RLOGF("this is fatal: %.2f", magic);
    }

#elif __LOGGING_BACKEND__ == 2
    #include <rclcpp/logging.hpp>
    #include <rclcpp/rclcpp.hpp>
    //* this should be a rclcpp::Node derived class *//
    #define RLOGD(...) RCLCPP_DEBUG(this->get_logger(), ##__VA_ARGS__)
    #define RLOGI(...) RCLCPP_INFO(this->get_logger(), ##__VA_ARGS__)
    #define RLOGW(...) RCLCPP_WARN(this->get_logger(), ##__VA_ARGS__)
    #define RLOGE(...) RCLCPP_ERROR(this->get_logger(), ##__VA_ARGS__)
    #define RLOGF(...) RCLCPP_FATAL(this->get_logger(), ##__VA_ARGS__)

    void __run_logger_test__(){
        class Ros2LoggerTest : public rclcpp::Node
        {
        public:
            Ros2LoggerTest() : Node("Ros2LoggerTest")
            {
                RLOGD("this is debug");
                RLOGI("this is info");
                RLOGW("this is warn");
                RLOGE("this is error");
                RLOGF("this is fatal");
            }
        };

        rclcpp::Node::SharedPtr _node = std::make_shared<Ros2LoggerTest>();
    }

#elif __LOGGING_BACKEND__ == 1
    #include <ros/ros.h>
    #define RLOGD(...) ROS_DEBUG(__VA_ARGS__)
    #define RLOGI(...) ROS_INFO(__VA_ARGS__)
    #define RLOGW(...) ROS_WARN(__VA_ARGS__)
    #define RLOGE(...) ROS_ERROR(__VA_ARGS__)
    #define RLOGF(...) ROS_FATAL(__VA_ARGS__)

    void __run_ros1_logger_test__(){
        RLOGD("this is debug");
        RLOGI("this is info");
        RLOGW("this is warn");
        RLOGE("this is error");
        RLOGF("this is fatal");
    }

#else
    #define RLOGD(...) LOGPF(__VA_ARGS__)
    #define RLOGI(...) LOGPF(__VA_ARGS__)
    #define RLOGW(...) LOGPF(__VA_ARGS__)
    #define RLOGE(...) LOGPF(__VA_ARGS__)
    #define RLOGF(...) LOGPF(__VA_ARGS__)
#endif

/** profiling utility */
static inline __attribute__((always_inline)) uint64_t gfGetCurrentMicros()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::time_point_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()).time_since_epoch()).count();
}

static inline std::shared_ptr<uint64_t> gfHangStopWatch(const char* func_name)
{
    uint64_t* pts_us = new uint64_t;
    *pts_us = gfGetCurrentMicros();
    RLOGD("stopwatch tick by %s", func_name);
    return std::shared_ptr<uint64_t>(pts_us, [func_name](uint64_t* ptr){
        uint64_t ts_us = gfGetCurrentMicros();
        RLOGI("stopwatch tock by %s, elapse: %ld us", func_name, (ts_us - *ptr));
        delete ptr;
    });
}

#define HANG_STOPWATCH() auto _ProfilingUtilsPtr_ = gfHangStopWatch(__FUNCTION__);

static inline void _print_ros_env_(){
    HANG_STOPWATCH()
    LOGPF("\n*** ROS_VERSION: %s ***\n", getenv("ROS_VERSION"));
    LOGPF("\n*** ROS_DISTRO: %s ***\n", getenv("ROS_DISTRO"));
    LOGPF("\n*** ROS_PYTHON_VERSION: %s ***\n", getenv("ROS_PYTHON_VERSION"));
}

#endif //__LOGGING_UTILS_H__