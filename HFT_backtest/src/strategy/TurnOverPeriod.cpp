#include "TurnOverPeriod.h"

namespace alphaone
{
void TurnOverPeriod::Update(const size_t tick, OrderReportMessageExecuted *o)
{
    const double sign{o->Side == OrderReportSide::Buy ? +1.0 : -1.0};
    const double origin_qty{static_cast<double>(o->Qty)};
    const double offset_qty{sign * position_ < 0.0 ? std::min(std::fabs(position_), origin_qty)
                                                   : 0.0};
    const double remain_qty{origin_qty - offset_qty};

    position_ += sign * origin_qty;

    // for closing
    turnover_ += offset_qty;
    tick_ += offset_qty * tick;

    // for opening
    tick_ -= remain_qty * tick;
}

size_t TurnOverPeriod::Settle(const size_t tick) const
{
    return std::floor(turnover_ == 0.0 ? 0.0 : (tick_ + std::fabs(position_) * tick) / turnover_);
}

}  // namespace alphaone
