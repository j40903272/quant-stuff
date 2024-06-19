#include "AdjustedWeightedTickIntervalCounter.h"

namespace alphaone
{
AdjustedWeightedTickIntervalCounter::AdjustedWeightedTickIntervalCounter(
    const Book *book, MultiBookManager *multi_book_manager, const nlohmann::json &spec)
    : Counter{book, multi_book_manager}
    , multiplier_{spec.value("multiplier", 0.1)}
    , last_adjusted_weighted_tick_price_{0.0}
{
    SetElements();
}

AdjustedWeightedTickIntervalCounter::~AdjustedWeightedTickIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void AdjustedWeightedTickIntervalCounter::OnPacketEnd(const Timestamp event_loop_time,
                                                      const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    const BookPrice bid_price{book_->GetPrice(BID)};
    const BookPrice ref_price{book_->GetWeightedPrice()};
    const BookPrice tick_size{book_->GetSymbol()->GetTickSize(bid_price, true)};
    const uint32_t  move{static_cast<uint32_t>(
        std::abs(ref_price - last_adjusted_weighted_tick_price_) / (multiplier_ * tick_size))};
    if (move > 0)
    {
        last_update_timestamp_             = event_loop_time;
        last_adjusted_weighted_tick_price_ = ref_price;
        count_ += move;
    }
}

void AdjustedWeightedTickIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    if (multiplier_ != 0.1)
        elements_.emplace_back(to_succinct_string(multiplier_));
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

std::string AdjustedWeightedTickIntervalCounter::Name() const
{
    return "AdjustedWeightedTickInterval";
}

void AdjustedWeightedTickIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_adjusted_weighted_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_adjusted_weighted_tick_price: {}", str,
                last_adjusted_weighted_tick_price_);
    SPDLOG_INFO("[{}] adjusted_weighted_ticks: {}", str, count_);
}
}  // namespace alphaone
