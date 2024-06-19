#ifndef _MULTICASTLISTENER_H_
#define _MULTICASTLISTENER_H_

namespace alphaone
{

class Timestamp;

class MulticastListener
{

  public:
    MulticastListener()                          = default;
    MulticastListener(const MulticastListener &) = delete;
    MulticastListener &operator=(const MulticastListener &) = delete;
    virtual ~MulticastListener()                            = default;

    virtual void OnMulticastPacket(const Timestamp &event_loop_time, void *ptr) = 0;
};


}  // namespace alphaone


#endif