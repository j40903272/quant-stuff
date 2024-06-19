#include "Profiler.h"

namespace alphaone
{

Profiler::Profiler(const std::string &name)
    : name_{name}, limit_event_ts_{Timestamp::invalid(), Timestamp::invalid()}
{
    for (int s = Start; s < StartEnd; ++s)
    {
        clock_ts_[s].reserve(PROFILER_BUFFER);
        event_ts_[s].reserve(PROFILER_BUFFER);
    }
}

Profiler::~Profiler()
{
    const auto &start_size{clock_ts_[Start].size()};
    const auto &end_size{clock_ts_[End].size()};
    if (start_size != end_size)
    {
        SPDLOG_ERROR("[{}] start size = {} is not equal to end size = {}, thus won't calculate",
                     name_, start_size, end_size);
        return;
    }
    auto max{INT64_MIN};
    auto min{INT64_MAX};
    auto event_size{event_ts_[Start].size()};
    for (size_t i = 0; i < start_size; ++i)
    {
        const auto d{(clock_ts_[End][i] - clock_ts_[Start][i]).count()};
        durations_.push_back(d);
        if (d > max)
        {
            max                  = d;
            limit_event_ts_[Max] = i < event_size ? event_ts_[Start][i] : Timestamp::invalid();
        }
        if (d < min)
        {
            min                  = d;
            limit_event_ts_[Min] = i < event_size ? event_ts_[Start][i] : Timestamp::invalid();
        }
        sum_duration_ += d;
        square_sum_duration_ += d * d;
    }
    const long double size{static_cast<long double>(start_size)};
    const auto        q1{start_size / 4};
    const auto        q2{start_size / 2};
    const auto        q3{q1 + q2};
    // this only needs O(N) instead of O(NlogN)
    std::nth_element(durations_.begin(), durations_.begin() + q1, durations_.end());
    std::nth_element(durations_.begin() + q1 + 1, durations_.begin() + q2, durations_.end());
    std::nth_element(durations_.begin() + q2 + 1, durations_.begin() + q3, durations_.end());
    const auto avg{sum_duration_ / size};
    const auto std{sqrt((square_sum_duration_ - size * avg * avg) / (size - 1))};
    SPDLOG_INFO("[{}] avg_elasped_time = {} ns, std = {} ns, called_times = {}", name_, avg, std,
                start_size);
    SPDLOG_INFO("[{}] min = {} ns, 25% = {} ns, 50% = {} ns, 75% = {} ns, max = {} ns", name_, min,
                durations_[q1], durations_[q2], durations_[q3], max);
    SPDLOG_INFO("[{}] min_ts = {}, max_ts = {}", name_, limit_event_ts_[Min], limit_event_ts_[Max]);
}

}  // namespace alphaone