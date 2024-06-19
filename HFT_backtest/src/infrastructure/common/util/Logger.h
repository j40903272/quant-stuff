#ifndef _LOGGER_H_
#define _LOGGER_H_

#include "infrastructure/common/util/Helper.h"
#include "infrastructure/common/util/LoggerFormat.h"

#define __class__ (alphaone::helper::demangle(typeid(this).name()))

// clang-format off
#define LOGGER(TIMESTAMP, LOGGERFORMAT, TEXT) (alphaone::logger((__class__), (__func__), (TIMESTAMP), (LOGGERFORMAT), (TEXT)))
// clang-format on

#endif
