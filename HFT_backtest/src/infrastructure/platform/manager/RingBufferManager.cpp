#include "RingBufferManager.h"

#include "infrastructure/common/spdlog/spdlog.h"

namespace alphaone
{

void RingBufferManager::AddRingBuffer(int index, int block, int packet, int length)
{
    ring_buffers_.emplace_back(index, block, packet, length, -1, 0);
    SPDLOG_INFO("Create RingBuffer index={}, block={}, packet_size={}, length={}", index, block,
                packet, length);
}

void RingBufferManager::AddRingBufferListener(RingBufferListener *l)
{
    if (std::find(listeners_.begin(), listeners_.end(), l) == listeners_.end())
        listeners_.push_back(l);
}

void RingBufferManager::ReserveBuffer(size_t length)
{
    ring_buffers_.reserve(length);
}

void RingBufferManager::Process(const Timestamp &event_loop_time)
{
    for (auto &rb : ring_buffers_)
    {
        while (rb.SequentialGet(&ptr_))
        {
            for (auto &l : listeners_)
                l->OnRingBufferNewData(event_loop_time, ptr_);
        }
    }
}

}  // namespace alphaone
