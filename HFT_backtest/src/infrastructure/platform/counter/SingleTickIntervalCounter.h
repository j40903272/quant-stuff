#ifndef _SINGLETICKINTERVALCOUNTER_H_
#define _SINGLETICKINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class SingleTickIntervalCounter : public Counter
{
  public:
    SingleTickIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    SingleTickIntervalCounter(const SingleTickIntervalCounter &) = delete;
    SingleTickIntervalCounter &operator=(const SingleTickIntervalCounter &) = delete;
    SingleTickIntervalCounter(SingleTickIntervalCounter &&)                 = delete;
    SingleTickIntervalCounter &operator=(SingleTickIntervalCounter &&) = delete;

    ~SingleTickIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
    BookPrice last_single_tick_bid_price_;
    BookPrice last_single_tick_ask_price_;
};
}  // namespace alphaone

#endif
