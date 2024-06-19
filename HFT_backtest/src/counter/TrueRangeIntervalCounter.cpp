#include "TrueRangeIntervalCounter.h"

#include "counter/CounterFactory.h"
#include "infrastructure/common/math/Math.h"

namespace alphaone
{
TrueRangeIntervalCounter::TrueRangeIntervalCounter(const ObjectManager *object_manager,
                                                   MultiBookManager *   multi_book_manager,
                                                   Engine *engine, const nlohmann::json &spec)
    : Counter{multi_book_manager->GetBook(spec["symbol"].get<std::string>()), multi_book_manager}
    , clock_time_counter_spec_(spec.value("ClockTimeInterval", nlohmann::json()))
    , clock_time_counter_{nullptr}
    , last_clock_time_count_{0UL}
    , tick_{spec.value("tick", 0.003)}
    , tick_inverse_{1.0 / tick_}
    , last_close_{NaN}
    , this_close_{NaN}
    , last_volatility_{0.0}
    , this_volatility_{0.0}
{
    clock_time_counter_spec_["symbol"] = spec["symbol"].get<std::string>();
    clock_time_counter_                = CounterFactory::RetrieveCounterFromCounterSpec(
        object_manager, multi_book_manager, engine, "ClockTimeInterval", clock_time_counter_spec_);
    SetElements();
}

TrueRangeIntervalCounter::~TrueRangeIntervalCounter()
{
    if (clock_time_counter_)
    {
        delete clock_time_counter_;
        clock_time_counter_ = nullptr;
    }
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TrueRangeIntervalCounter::OnTrade(const Timestamp             event_loop_time,
                                       const BookDataMessageTrade *o)
{
    // store latest trade price as close
    this_close_ = o->GetTradePrice();
}
void TrueRangeIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                           const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(!IsPrepared()))
    {
        return;
    }

    if (last_clock_time_count_ == clock_time_counter_->GetCount())
    {
        return;
    }
    last_clock_time_count_ = clock_time_counter_->GetCount();

    if (BRANCH_LIKELY(std::isnan(last_close_)))
    {
        last_close_ = this_close_;
        return;
    }

    this_volatility_ += std::abs(y_log(this_close_ / last_close_));
    last_close_ = this_close_;
    SPDLOG_DEBUG("{} {} {:.5f} {:.5f}", event_loop_time, this_close_, this_volatility_,
                 last_volatility_);

    const uint32_t flip{
        static_cast<uint32_t>(std::floor((this_volatility_ - last_volatility_) * tick_inverse_))};
    if (flip > 0)
    {
        last_update_timestamp_ = event_loop_time;
        last_volatility_       = this_volatility_;
        count_ += flip;
    }
}

void TrueRangeIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(to_succinct_string(tick_));
    elements_.emplace_back(clock_time_counter_->ToString());
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

std::string TrueRangeIntervalCounter::Name() const
{
    return "TrueRangeInterval";
}

void TrueRangeIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_volatility_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] volatility_ticks: {}", str, count_);
}

void TrueRangeIntervalCounter::WarmUp()
{
    warmed_up_ = true;

    if (clock_time_counter_ != nullptr)
    {
        clock_time_counter_->WarmUp();
    }

    SubscribeBook(GetBook());
}

}  // namespace alphaone
