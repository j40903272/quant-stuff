#include "DoubleTickIntervalCounter.h"

namespace alphaone
{
DoubleTickIntervalCounter::DoubleTickIntervalCounter(const Book *      book,
                                                     MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
    , last_double_tick_bid_price_{NaN}
    , last_double_tick_ask_price_{NaN}
{
    SetElements();
}

DoubleTickIntervalCounter::~DoubleTickIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void DoubleTickIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                            const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    const BookPrice bid_price{book_->GetPrice(BID)};
    const BookPrice ask_price{book_->GetPrice(ASK)};

    if (BRANCH_UNLIKELY(std::isnan(last_double_tick_bid_price_) or
                        std::isnan(last_double_tick_ask_price_)))
    {
        last_double_tick_bid_price_ = bid_price;
        last_double_tick_ask_price_ = ask_price;
        return;
    }

    const bool double_inward{bid_price > last_double_tick_bid_price_ &&
                             ask_price < last_double_tick_ask_price_};
    if (double_inward)
    {
        last_double_tick_bid_price_ = bid_price;
        last_double_tick_ask_price_ = ask_price;
        return;
    }

    const bool double_upward{bid_price > last_double_tick_bid_price_ &&
                             ask_price > last_double_tick_ask_price_};
    const bool double_downward{bid_price < last_double_tick_bid_price_ &&
                               ask_price < last_double_tick_ask_price_};
    if (double_upward or double_downward)
    {
        last_update_timestamp_      = event_loop_time;
        last_double_tick_bid_price_ = bid_price;
        last_double_tick_ask_price_ = ask_price;
        count_ += 1;
    }
}

std::string DoubleTickIntervalCounter::Name() const
{
    return "DoubleTickInterval";
}

void DoubleTickIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_double_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_double_tick_bid_price: {}", str, last_double_tick_bid_price_);
    SPDLOG_INFO("[{}] last_double_tick_ask_price: {}", str, last_double_tick_ask_price_);
    SPDLOG_INFO("[{}] double_ticks: {}", str, count_);
}
}  // namespace alphaone
