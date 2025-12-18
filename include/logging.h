#include <sstream>
#include <string>
#include <unistd.h>

#pragma once

namespace GNNPro_lib{
namespace common{

enum class LogLevel {TRACE, DEBUG, INFO, WARNING, ERROR, FATAL};

#define CHECK(x) \
    if (!(x)) LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << " "

static const char *LOG_LEVELS[] = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};

class LogMessage : public std::basic_ostringstream<char>{
public:
    LogMessage(const char* fname, int line, LogLevel severity);
    ~LogMessage();

protected:
    void GenerateLogMessage(bool log_time);

private:
    const char* fname_;
    int line_;
    LogLevel severity_;
};

class LogMessageFatal : public LogMessage{
public:
    LogMessageFatal(const char* file, int line);
    ~LogMessageFatal();
};

#define _LOG_TRACE    LogMessage(__FILE__, __LINE__, LogLevel::TRACE)
#define _LOG_DEBUG    LogMessage(__FILE__, __LINE__, LogLevel::DEBUG)
#define _LOG_INFO     LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _LOG_WARNING  LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _LOG_ERROR    LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _LOG_FATAL    LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _LOG_##severity
#define LOG(...) _LOG(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFlagFromEnv();

}   //namespace common
}   //namescpce GNNPro_lib  