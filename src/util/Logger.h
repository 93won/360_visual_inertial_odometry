/**
 * @file      Logger.h
 * @brief     Logging utility using spdlog
 * @author    360-VIO Team
 * @date      2025-11-25
 */

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>

namespace vio_360 {

class Logger {
public:
    static void Init() {
        if (s_initialized) return;
        
        auto console = spdlog::stdout_color_mt("vio");
        console->set_pattern("%^[%L]%$ %v");
        console->set_level(spdlog::level::info);
        spdlog::set_default_logger(console);
        s_initialized = true;
    }
    
    static void SetLevel(spdlog::level::level_enum level) {
        spdlog::set_level(level);
    }
    
    template<typename... Args>
    static void Info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        Init();
        spdlog::info(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void Warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        Init();
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void Error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        Init();
        spdlog::error(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void Debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        Init();
        spdlog::debug(fmt, std::forward<Args>(args)...);
    }
    
    // Print raw line without log level prefix
    static void Print(const std::string& msg) {
        Init();
        spdlog::info("{}", msg);
    }

private:
    static inline bool s_initialized = false;
};

// Convenience macros
#define LOG_INFO(...) vio_360::Logger::Info(__VA_ARGS__)
#define LOG_WARN(...) vio_360::Logger::Warn(__VA_ARGS__)
#define LOG_ERROR(...) vio_360::Logger::Error(__VA_ARGS__)
#define LOG_DEBUG(...) vio_360::Logger::Debug(__VA_ARGS__)

} // namespace vio_360
