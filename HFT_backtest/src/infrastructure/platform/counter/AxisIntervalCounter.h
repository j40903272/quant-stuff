#ifndef _AXISINTERVALCOUNTER_H_
#define _AXISINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class AxisIntervalCounter : public Counter
{
  public:
    AxisIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    AxisIntervalCounter(const AxisIntervalCounter &) = delete;
    AxisIntervalCounter &operator=(const AxisIntervalCounter &) = delete;
    AxisIntervalCounter(AxisIntervalCounter &&)                 = delete;
    AxisIntervalCounter &operator=(AxisIntervalCounter &&) = delete;

    ~AxisIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    BookPrice last_axis_price_;
};
}  // namespace alphaone

#endif
