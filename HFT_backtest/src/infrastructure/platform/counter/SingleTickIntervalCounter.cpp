#include "SingleTickIntervalCounter.h"

namespace alphaone
{
SingleTickIntervalCounter::SingleTickIntervalCounter(const Book *      book,
                                                     MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
    , last_single_tick_bid_price_{NaN}
    , last_single_tick_ask_price_{NaN}
{
    SetElements();
}

SingleTickIntervalCounter::~SingleTickIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void SingleTickIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                            const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    const BookPrice bid_price{book_->GetPrice(BID)};
    const BookPrice ask_price{book_->GetPrice(ASK)};

    if (BRANCH_UNLIKELY(std::isnan(last_single_tick_bid_price_) or
                        std::isnan(last_single_tick_ask_price_)))
    {
        last_single_tick_bid_price_ = bid_price;
        last_single_tick_ask_price_ = ask_price;
        return;
    }

    if (last_single_tick_bid_price_ != bid_price or last_single_tick_ask_price_ != ask_price)
    {
        last_update_timestamp_      = event_loop_time;
        last_single_tick_bid_price_ = bid_price;
        last_single_tick_ask_price_ = ask_price;
        count_ += 1;
    }
}

std::string SingleTickIntervalCounter::Name() const
{
    return "SingleTickInterval";
}

void SingleTickIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_single_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_single_tick_bid_price: {}", str, last_single_tick_bid_price_);
    SPDLOG_INFO("[{}] last_single_tick_ask_price: {}", str, last_single_tick_ask_price_);
    SPDLOG_INFO("[{}] single_ticks: {}", str, count_);
}
}  // namespace alphaone
