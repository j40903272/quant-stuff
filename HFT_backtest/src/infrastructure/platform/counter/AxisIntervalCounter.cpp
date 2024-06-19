#include "AxisIntervalCounter.h"

namespace alphaone
{
AxisIntervalCounter::AxisIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}, last_axis_price_{NaN}
{
    SetElements();
}

AxisIntervalCounter::~AxisIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void AxisIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                      const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    const BookPrice bid_price{book_->GetPrice(BID)};
    const BookPrice ask_price{book_->GetPrice(ASK)};
    if (BRANCH_UNLIKELY(std::isnan(last_axis_price_)))
    {
        last_axis_price_ = bid_price;
        return;
    }

    if (last_axis_price_ < bid_price)
    {
        last_update_timestamp_ = event_loop_time;
        last_axis_price_       = bid_price;
        count_ += 1;
    }
    else if (last_axis_price_ > ask_price)
    {
        last_update_timestamp_ = event_loop_time;
        last_axis_price_       = ask_price;
        count_ += 1;
    }
}

std::string AxisIntervalCounter::Name() const
{
    return "AxisInterval";
}

void AxisIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_axis_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_axis_price: {}", str, last_axis_price_);
    SPDLOG_INFO("[{}] axes: {}", str, count_);
}
}  // namespace alphaone
