#include "MulticastReceiverManager.h"

#include "infrastructure/common/spdlog/spdlog.h"

#include <algorithm>

namespace alphaone
{

void MulticastReceiverManager::AddMulticastReceiver(const char *multicast_ip, uint16_t port,
                                                    const char *interface_ip)
{
    receivers_.emplace_back(multicast_ip, port, interface_ip);
    SPDLOG_INFO("Create MulticastReceiver at {}:{} over interface {}", multicast_ip, port,
                interface_ip);
}
void MulticastReceiverManager::AddMulticastListener(MulticastListener *l)
{
    if (std::find(listeners_.begin(), listeners_.end(), l) == listeners_.end())
        listeners_.push_back(l);
}

void MulticastReceiverManager::ReserveBuffer(size_t length)
{
    receivers_.reserve(length);
}

void MulticastReceiverManager::Process(const Timestamp &event_loop_time)
{
    for (auto &r : receivers_)
    {
        ptr_ = r.Receive();
        while (ptr_)
        {
            for (auto &l : listeners_)
                l->OnMulticastPacket(event_loop_time, ptr_);
            ptr_ = r.Receive();
        }
    }
}


}  // namespace alphaone
