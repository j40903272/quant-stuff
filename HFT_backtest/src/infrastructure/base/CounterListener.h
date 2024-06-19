#ifndef _COUNTERLISTENER_H_
#define _COUNTERLISTENER_H_

#include "infrastructure/common/datetime/Timestamp.h"

namespace alphaone
{
class Counter;

class CounterListener
{
  public:
    CounterListener()                        = default;
    CounterListener(const CounterListener &) = delete;
    CounterListener &operator=(const CounterListener &) = delete;

    virtual ~CounterListener() = default;

    virtual void OnSnapshot(const Timestamp &event_loop_time) = 0;
    virtual void OnSignal(const Timestamp &event_loop_time)   = 0;
};
}  // namespace alphaone
#endif
