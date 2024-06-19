#include "TradeIntervalCounter.h"

namespace alphaone
{
TradeIntervalCounter::TradeIntervalCounter(const Book *book, MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

TradeIntervalCounter::~TradeIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TradeIntervalCounter::OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    last_update_timestamp_ = event_loop_time;
    count_ += 1;
}

std::string TradeIntervalCounter::Name() const
{
    return "TradeInterval";
}

void TradeIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_trade_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] trades: {}", str, count_);
}
}  // namespace alphaone
