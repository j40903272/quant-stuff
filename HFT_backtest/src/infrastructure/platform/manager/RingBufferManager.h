#ifndef _RINGBUFFERMANAGER_H_
#define _RINGBUFFERMANAGER_H_

#include "infrastructure/base/RingBufferListener.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/twone/ringbuffer/RingBuffer.h"

#include <vector>

namespace alphaone
{

class RingBufferManager
{

  public:
    RingBufferManager()  = default;
    ~RingBufferManager() = default;

    void AddRingBuffer(int index, int block, int packet, int length);
    void AddRingBufferListener(RingBufferListener *l);
    void ReserveBuffer(size_t length);

    void Process(const Timestamp &event_loop_time);

  private:
    void *ptr_{nullptr};

    std::vector<twone::RingBuffer>    ring_buffers_;
    std::vector<RingBufferListener *> listeners_;
};

}  // namespace alphaone


#endif