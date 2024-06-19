#ifndef _DOUBLETICKINTERVALCOUNTER_H_
#define _DOUBLETICKINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class DoubleTickIntervalCounter : public Counter
{
  public:
    DoubleTickIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    DoubleTickIntervalCounter(const DoubleTickIntervalCounter &) = delete;
    DoubleTickIntervalCounter &operator=(const DoubleTickIntervalCounter &) = delete;
    DoubleTickIntervalCounter(DoubleTickIntervalCounter &&)                 = delete;
    DoubleTickIntervalCounter &operator=(DoubleTickIntervalCounter &&) = delete;

    ~DoubleTickIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    BookPrice last_double_tick_bid_price_;
    BookPrice last_double_tick_ask_price_;
};
}  // namespace alphaone

#endif
