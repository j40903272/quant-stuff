#include "TimeIntervalCounter.h"

namespace alphaone
{
TimeIntervalCounter::TimeIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                                         const nlohmann::json &spec)
    : Counter{book, multi_book_manager}, threshold_in_second_{spec.value("duration_in_second", 1.0)}
{
    SetElements();
}

TimeIntervalCounter::~TimeIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}

void TimeIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                      const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (BRANCH_LIKELY(last_update_timestamp_.is_valid()))
    {
        if (const size_t tick{static_cast<size_t>(std::floor(
                (event_loop_time - last_update_timestamp_).to_double() / threshold_in_second_))};
            tick > 0)
        {
            last_update_timestamp_ += Duration::from_sec(tick * threshold_in_second_);
            count_ += tick;
        }
    }
    else
    {
        last_update_timestamp_ = event_loop_time;
        count_ += 1;
    }
}

void TimeIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(to_succinct_string(threshold_in_second_));
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

std::string TimeIntervalCounter::Name() const
{
    return "TimeInterval";
}

void TimeIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_time_tick_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] time_tick_count: {}", str, count_);
}
}  // namespace alphaone
