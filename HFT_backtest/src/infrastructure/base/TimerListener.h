#ifndef _TIMERLISTENER_H_
#define _TIMERLISTENER_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{
class TimerListener
{
  public:
    TimerListener()                      = default;
    TimerListener(const TimerListener &) = delete;
    TimerListener &operator=(const TimerListener &) = delete;

    ~TimerListener() = default;

    virtual void Timer(const Timestamp event_loop_time, const Timestamp call_back_time,
                       void *structure) = 0;
};
}  // namespace alphaone

#endif
