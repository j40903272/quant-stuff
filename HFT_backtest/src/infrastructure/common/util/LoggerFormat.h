#ifndef _LOGGERFORMAT_H_
#define _LOGGERFORMAT_H_

#include "infrastructure/common/datetime/Timestamp.h"

#include <iostream>
#include <thread>

namespace alphaone
{
enum class LoggerFormat
{
    Invalid     = 0,
    Information = 1,
    Warning     = 2,
    Error       = 3
};

std::string LoggerFormatToString(LoggerFormat format);
void        logger(std::string class_name, std::string function_name, Timestamp timestamp,
                   LoggerFormat format, const std::string &text);
}  // namespace alphaone

#endif
