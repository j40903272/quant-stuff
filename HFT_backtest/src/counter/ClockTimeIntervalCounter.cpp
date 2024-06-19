#include "ClockTimeIntervalCounter.h"

#include "counter/CounterFactory.h"
namespace alphaone
{
ClockTimeIntervalCounter::ClockTimeIntervalCounter(const Book *      book,
                                                   MultiBookManager *multi_book_manager,
                                                   Engine *engine, const nlohmann::json &spec)
    : Counter{book, multi_book_manager}
    , duration_{Duration::from_sec(spec.value("duration", 60.))}
    , time_start_{Timestamp::from_date_time(engine->GetDate(),
                                            spec.value("time_start", "09:00:00.000").c_str())}
    , time_end_{Timestamp::from_date_time(engine->GetDate(),
                                          spec.value("time_end", "13:26:00.000").c_str())}
{
    last_update_timestamp_ = time_start_;
    SetElements();
}

ClockTimeIntervalCounter::~ClockTimeIntervalCounter()
{
    if (IsWarmedUp())
    {
        Dump();
    }
}


void ClockTimeIntervalCounter::OnPacketEnd(const Timestamp                 event_loop_time,
                                           const BookDataMessagePacketEnd *o)
{
    if (BRANCH_UNLIKELY(not IsPrepared()))
    {
        return;
    }

    if (BRANCH_UNLIKELY(event_loop_time > time_end_))
    {
        return;
    }

    if (event_loop_time - last_update_timestamp_ > duration_)
    {
        while (last_update_timestamp_ + duration_ <= event_loop_time)
        {
            last_update_timestamp_ += duration_;
            ++count_;
        }
    }
}

std::string ClockTimeIntervalCounter::Name() const
{
    return "ClockTimerInterval";
}

void ClockTimeIntervalCounter::DumpDetail() const
{
    const auto &str = ToString();
    SPDLOG_INFO("[{}] last_clock_timestamp: {}", str, last_update_timestamp_);
    SPDLOG_INFO("[{}] events: {}", str, count_);
}

void ClockTimeIntervalCounter::SetElements()
{
    elements_.emplace_back(Name());
    elements_.emplace_back(to_succinct_string(duration_.to_sec()));
    elements_.emplace_back(time_start_.to_string());
    elements_.emplace_back(time_end_.to_string());
    if (symbol_ != nullptr)
        elements_.emplace_back(symbol_->GetRepresentativePid());
}

}  // namespace alphaone
