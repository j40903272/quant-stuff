#include "WeightedTickIntervalCounter.h"

#include "infrastructure/common/math/Math.h"

namespace alphaone
{
WeightedTickIntervalCounter::WeightedTickIntervalCounter(const Book *          book,
                                                         MultiBookManager *    multi_book_manager,
                                                         const nlohmann::json &spec)
    : Counter{book, multi_book_manager}
    , price_type_{(spec.value("price_type", "MidPrice") == "MidPrice") ? MidPrice : WeightedPrice}
    , accuracy_{spec.value("accuracy", 1.0)}
    , tick_{spec.value("tick", 0.0005)}
    , tick_inverse_{accuracy_ / tick_}
    , last_price_{NaN}
    , this_price_{NaN}
    , last_path_{0.0}
    , this_path_{0.0}
{
    if (accuracy_ < 1.0)
    {
        throw std::invalid_argument(
            fmt::format("accuracy={} should be no less than 1.0", accuracy_));
    }
    SetElements();
}

WeightedTickIntervalCounter::~WeightedTickIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void WeightedTickIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                              const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(!IsPrepared()))
    {
        return;
    }

    this_price_ = (price_type_ == MidPrice) ? book_->GetMidPrice() : book_->GetWeightedPrice();
    if (BRANCH_LIKELY(!std::isnan(last_price_)))
    {
        this_path_ += std::abs(y_log(this_price_ / last_price_));
    }
    last_price_ = this_price_;

    if (this_path_ > last_path_)
    {
        last_update_timestamp_ = event_loop_time;
        last_path_             = this_path_;
        count_                 = static_cast<uint32_t>(std::floor(this_path_ * tick_inverse_));
    }
}

void WeightedTickIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(price_type_ == MidPrice ? "MidPrice" : "WeightedPrice");
    elements_.emplace_back(to_succinct_string(tick_));
    if (accuracy_ != 1.0)
        elements_.emplace_back(to_succinct_string(accuracy_));
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

std::string WeightedTickIntervalCounter::Name() const
{
    return "WeightedTickInterval";
}

void WeightedTickIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_weighted_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] last_weighted_tick_price: {}", str, last_price_);
    SPDLOG_INFO("[{}] weighted_ticks: {}", str, count_);
}
}  // namespace alphaone
