#ifndef _TIMERSOURCE_H_
#define _TIMERSOURCE_H_

#include "infrastructure/base/TimerListener.h"
#include "infrastructure/common/datetime/Duration.h"
#include "infrastructure/common/datetime/Timestamp.h"

#include <functional>

namespace alphaone
{
class TimerSource
{
  public:
    TimerSource();
    TimerSource(const TimerSource &) = delete;
    TimerSource &operator=(const TimerSource &) = delete;

    virtual ~TimerSource() = default;

    // add a listener
    void AddTimerListener(TimerListener *listener, void *structure);

    template <typename X, typename Y>
    void AddTimerListener(X callback_function, Y pointer_callback_class, void *structure);

    // schedule events
    // each timer source can only handle one type of event scheduling, i.e. calling another
    // schedule() function overrides the previous one
    void Schedule(const Timestamp &start_time, const Duration &periodicity,
                  const Timestamp &end_time);
    void Schedule(const Timestamp &start_time, const Duration &periodicity,
                  const uint16_t number_of_times);
    void Schedule(const Timestamp &exact_time);

    // return Timestamp::invalid() if no events or done
    virtual const Timestamp PeekTimestamp();

    // give control to timer source and source will callback listener
    virtual void Process(const Timestamp event_loop_time);

  private:
    Timestamp timestamp_start_;
    Timestamp timestamp_end_;
    Duration  duration_periodicity_;

    std::vector<std::pair<std::function<void(const Timestamp, const Timestamp, void *)>, void *>>
        listeners_;
};

template <typename F, typename R>
void alphaone::TimerSource::AddTimerListener(F f, R r, void *structure)
{
    listeners_.push_back(std::make_pair(
        std::bind(f, r, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        structure));
}
}  // namespace alphaone

#endif
