#ifndef _RINGBUFFERLISTENER_H_
#define _RINGBUFFERLISTENER_H_

namespace alphaone
{

class Timestamp;

class RingBufferListener
{
  public:
    RingBufferListener()                           = default;
    RingBufferListener(const RingBufferListener &) = delete;
    RingBufferListener &operator=(const RingBufferListener &) = delete;
    virtual ~RingBufferListener()                             = default;

    virtual void OnRingBufferNewData(const Timestamp &event_loop_time, void *ptr) = 0;
};

}  // namespace alphaone


#endif