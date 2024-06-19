#include "TimerSource.h"

namespace alphaone
{
TimerSource::TimerSource()
    : timestamp_start_{Timestamp::invalid()}
    , timestamp_end_{Timestamp::invalid()}
    , duration_periodicity_{Duration::zero()}
{
}

void TimerSource::AddTimerListener(TimerListener *listener, void *structure)
{
    listeners_.push_back(
        std::make_pair(std::bind(&TimerListener::Timer, listener, std::placeholders::_1,
                                 std::placeholders::_2, std::placeholders::_3),
                       structure));
}

void TimerSource::Schedule(const Timestamp &start_time, const Duration &periodicity,
                           const Timestamp &end_time)
{
    timestamp_start_      = start_time;
    duration_periodicity_ = periodicity;
    timestamp_end_        = end_time;
}

void TimerSource::Schedule(const Timestamp &start_time, const Duration &periodicity,
                           const uint16_t number_of_times)
{
    timestamp_start_      = start_time;
    duration_periodicity_ = periodicity;
    timestamp_end_        = start_time + (periodicity * number_of_times);
}

void TimerSource::Schedule(const Timestamp &exact_time)
{
    timestamp_start_      = exact_time;
    duration_periodicity_ = Duration::zero();
    timestamp_end_        = Timestamp::invalid();
}

const Timestamp TimerSource::PeekTimestamp()
{
    return timestamp_start_;
}

void TimerSource::Process(const Timestamp event_loop_time)
{
    if (!timestamp_start_.is_valid())
    {
        return;
    }

    while (timestamp_start_ <= event_loop_time)
    {
        for (auto &listener : listeners_)
        {
            (listener.first)(event_loop_time, timestamp_start_, listener.second);
        }

        if (duration_periodicity_.is_greater_than_zero())
        {
            timestamp_start_ += duration_periodicity_;
            if (timestamp_start_ > timestamp_end_)
            {
                timestamp_start_ = Timestamp::invalid();
                return;
            }
        }
        else
        {
            timestamp_start_ = Timestamp::invalid();
            return;
        }
    }
}
}  // namespace alphaone
