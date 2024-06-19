#include "VolatilityIntervalCounter.h"

#include "infrastructure/common/math/Math.h"

namespace alphaone
{
VolatilityIntervalCounter::VolatilityIntervalCounter(const Book *          book,
                                                     MultiBookManager *    multi_book_manager,
                                                     const nlohmann::json &spec)
    : Counter{book, multi_book_manager}
    , symbol_{book_->GetSymbol()}
    , price_type_{(spec.value("price_type", "MidPrice") == "MidPrice") ? MidPrice : WeightedPrice}
    , accuracy_{spec.value("accuracy", 1.0)}
    , given_tick_{spec.value("tick", 0.0005)}
    , tick_{given_tick_ == 0.0 && symbol_ != nullptr
                ? symbol_->GetTickSize(symbol_->GetReferencePrice(), false) /
                      symbol_->GetReferencePrice() / accuracy_
                : given_tick_}
    , tick_squared_inverse_{accuracy_ / square(tick_)}
    , last_price_{NaN}
    , this_price_{NaN}
    , last_volatility_{0.0}
    , this_volatility_{0.0}
{
    if (accuracy_ < 1.0)
    {
        throw std::invalid_argument(
            fmt::format("accuracy={} should be no less than 1.0", accuracy_));
    }
    if (tick_ == 0.0)
    {
        throw std::invalid_argument(fmt::format("tick={} should be positive", tick_));
    }
    SetElements();
}

VolatilityIntervalCounter::~VolatilityIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void VolatilityIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                            const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(!IsPrepared()))
    {
        return;
    }

    this_price_ = (price_type_ == MidPrice) ? book_->GetMidPrice() : book_->GetWeightedPrice();
    if (BRANCH_LIKELY(!std::isnan(last_price_)))
    {
        this_volatility_ += square(y_log(this_price_ / last_price_));
    }
    last_price_ = this_price_;

    if (this_volatility_ > last_volatility_)
    {
        last_update_timestamp_ = event_loop_time;
        last_volatility_       = this_volatility_;
        count_ = static_cast<uint32_t>(std::floor(this_volatility_ * tick_squared_inverse_));
    }
}

void VolatilityIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(price_type_ == MidPrice ? "MidPrice" : "WeightedPrice");
    elements_.emplace_back(to_succinct_string(given_tick_));
    if (accuracy_ != 1.0)
    {
        elements_.emplace_back(to_succinct_string(accuracy_));
    }
    if (symbol_ != nullptr)
    {
        elements_.emplace_back(symbol_->GetRepresentativePid());
    }
}

std::string VolatilityIntervalCounter::Name() const
{
    return "VolatilityInterval";
}

void VolatilityIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_volatility_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] volatility_ticks: {}", str, count_);
}
}  // namespace alphaone
