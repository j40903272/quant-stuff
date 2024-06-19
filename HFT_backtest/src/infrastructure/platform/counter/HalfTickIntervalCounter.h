#ifndef _HALFTICKINTERVALCOUNTER_H_
#define _HALFTICKINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class HalfTickIntervalCounter : public Counter
{
  public:
    HalfTickIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    HalfTickIntervalCounter(const HalfTickIntervalCounter &) = delete;
    HalfTickIntervalCounter &operator=(const HalfTickIntervalCounter &) = delete;
    HalfTickIntervalCounter(HalfTickIntervalCounter &&)                 = delete;
    HalfTickIntervalCounter &operator=(HalfTickIntervalCounter &&) = delete;

    ~HalfTickIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    BookPrice last_half_tick_bid_price_;
    BookPrice last_half_tick_ask_price_;
};
}  // namespace alphaone

#endif
