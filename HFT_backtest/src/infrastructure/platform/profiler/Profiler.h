#ifndef _PROFILER_H_
#define _PROFILER_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/spdlog/spdlog.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace alphaone
{

#define TIME_START()                                                                               \
    static Profiler _p{__class__ + "::" + __func__};                                               \
    _p.Stamp<StartEndSide::Start>()
#define TIME_START_WITH_TS(t)                                                                      \
    static Profiler _p{__class__ + "::" + __func__};                                               \
    _p.Stamp<StartEndSide::Start>(t)
#define TIME_END() _p.Stamp<StartEndSide::End>()
#define TIME_END_RETURN()                                                                          \
    TIME_END();                                                                                    \
    return

#define PROFILER_BUFFER 1024768 * 8

enum StartEndSide
{
    Start    = 0,
    End      = 1,
    StartEnd = 2,
};

enum MinMax
{
    Min    = 0,
    Max    = 1,
    MinMax = 2,
};

using Clock          = std::chrono::high_resolution_clock;
using ClockTimestamp = Clock::time_point;


class Profiler
{
  public:
    Profiler(const std::string &name);
    ~Profiler();
    template <StartEndSide side>
    inline void Stamp()
    {
        clock_ts_[side].emplace_back(Clock::now());
    }
    template <StartEndSide side>
    inline void Stamp(const Timestamp &event_loop_time)
    {
        clock_ts_[side].emplace_back(Clock::now());
        event_ts_[side].emplace_back(event_loop_time);
    }


  private:
    std::vector<ClockTimestamp> clock_ts_[StartEnd];
    std::vector<Timestamp>      event_ts_[StartEnd];
    std::vector<int64_t>        durations_;
    std::string                 name_;
    int64_t                     sum_duration_;
    int64_t                     square_sum_duration_;
    size_t                      index_;
    Timestamp                   limit_event_ts_[MinMax];
};

}  // namespace alphaone


#endif