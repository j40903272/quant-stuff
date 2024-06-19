#include "TradeThroughIntervalCounter.h"

namespace alphaone
{
TradeThroughIntervalCounter::TradeThroughIntervalCounter(const Book *      book,
                                                         MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
{
    SetElements();
}

TradeThroughIntervalCounter::~TradeThroughIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TradeThroughIntervalCounter::OnTrade(const Timestamp             event_loop_time,
                                          const BookDataMessageTrade *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (const BookPrice price{o->GetTradePrice()};
        price > book_->GetLastTouchPrice(ASK) or price < book_->GetLastTouchPrice(BID))
    {
        last_update_timestamp_ = event_loop_time;
        count_ += 1;
    }
}

std::string TradeThroughIntervalCounter::Name() const
{
    return "TradeThroughInterval";
}

void TradeThroughIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_trade_through_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] trade_throughs: {}", str, count_);
}
}  // namespace alphaone
