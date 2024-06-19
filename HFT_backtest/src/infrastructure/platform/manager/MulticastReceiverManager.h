#ifndef _MULTICASTRECEIVERMANAGER_H_
#define _MULTICASTRECEIVERMANAGER_H_

#include "infrastructure/base/MulticastListener.h"
#include "infrastructure/platform/receiver/MulticastReceiver.h"

#include <vector>

namespace alphaone
{

class MulticastReceiverManager
{

  public:
    MulticastReceiverManager()  = default;
    ~MulticastReceiverManager() = default;

    void AddMulticastReceiver(const char *multicast_ip, uint16_t port, const char *interface_ip);
    void AddMulticastListener(MulticastListener *l);
    void ReserveBuffer(size_t length);

    void Process(const Timestamp &event_loop_time);

  private:
    void *ptr_{nullptr};

    std::vector<MulticastReceiver>   receivers_;
    std::vector<MulticastListener *> listeners_;
};

}  // namespace alphaone


#endif