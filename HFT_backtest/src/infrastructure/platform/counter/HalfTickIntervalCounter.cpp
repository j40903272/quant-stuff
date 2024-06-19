#include "HalfTickIntervalCounter.h"

namespace alphaone
{
HalfTickIntervalCounter::HalfTickIntervalCounter(const Book *      book,
                                                 MultiBookManager *multi_book_manager)
    : Counter{book, multi_book_manager}
    , last_half_tick_bid_price_{NaN}
    , last_half_tick_ask_price_{NaN}
{
    SetElements();
}

HalfTickIntervalCounter::~HalfTickIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void HalfTickIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                          const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    const BookPrice bid_price{book_->GetPrice(BID)};
    const BookPrice ask_price{book_->GetPrice(ASK)};

    if (BRANCH_UNLIKELY(std::isnan(last_half_tick_bid_price_) or
                        std::isnan(last_half_tick_ask_price_)))
    {
        last_half_tick_bid_price_ = bid_price;
        last_half_tick_ask_price_ = ask_price;
        return;
    }

    const BookPrice tick_size{book_->GetSymbol()->GetTickSize(bid_price, true)};
    const BookPrice half_price_prev{last_half_tick_bid_price_ + last_half_tick_ask_price_};
    const BookPrice half_price_this{bid_price + ask_price};
    const uint32_t  half{static_cast<uint32_t>(
        std::floor(0.5 + std::abs(half_price_this - half_price_prev) / tick_size))};
    if (half > 0)
    {
        last_update_timestamp_    = event_loop_time;
        last_half_tick_bid_price_ = bid_price;
        last_half_tick_ask_price_ = ask_price;
        count_ += half;
    }
}

std::string HalfTickIntervalCounter::Name() const
{
    return "HalfTickInterval";
}

void HalfTickIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_half_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_half_tick_bid_price: {}", str, last_half_tick_bid_price_);
    SPDLOG_INFO("[{}] last_half_tick_ask_price: {}", str, last_half_tick_ask_price_);
    SPDLOG_INFO("[{}] half_ticks: {}", str, count_);
}
}  // namespace alphaone
