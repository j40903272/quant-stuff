#ifndef _TURNOVERPERIOD_H_
#define _TURNOVERPERIOD_H_

#include "infrastructure/base/OrderReportMessage.h"

#include <cmath>

namespace alphaone
{
struct TurnOverPeriod
{
    void   Update(const size_t tick, OrderReportMessageExecuted *o);
    size_t Settle(const size_t tick) const;

    double turnover_{0.0};
    double position_{0.0};
    double tick_{0.0};
};
}  // namespace alphaone

#endif
