#include "infrastructure/common/util/LoggerFormat.h"

#include <iostream>
#include <thread>

namespace alphaone
{
std::string LoggerFormatToString(LoggerFormat format)
{
    switch (format)
    {
    case LoggerFormat::Information:
        return "\033[1m\033[32mINFORMATION\033[0m";
    case LoggerFormat::Warning:
        return "\033[1m\033[33mWARNING\033[0m";
    case LoggerFormat::Error:
        return "\033[1m\033[31mERROR\033[0m";
    default:
        return "INVALID";
    }
}

void logger(std::string class_name, std::string function_name, Timestamp timestamp,
            LoggerFormat format, const std::string &text)
{
    std::stringstream ss;
    if (timestamp.is_valid())
    {
        ss << timestamp << " ";
    }

    // clang-format off
    ss
        << "[" << class_name << "::" << function_name << "]"
        << " " << LoggerFormatToString(format)
        << " " << text
        << std::endl;
    // clang-format on

    if (format == LoggerFormat::Error)
    {
        std::cerr << ss.str();
    }
    else
    {
        std::cout << ss.str();
    }
}
}  // namespace alphaone
