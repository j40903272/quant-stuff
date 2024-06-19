#ifndef _TRADEINTERVALCOUNTER_H_
#define _TRADEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class TradeIntervalCounter : public Counter
{
  public:
    TradeIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    TradeIntervalCounter(const TradeIntervalCounter &) = delete;
    TradeIntervalCounter &operator=(const TradeIntervalCounter &) = delete;
    TradeIntervalCounter(TradeIntervalCounter &&)                 = delete;
    TradeIntervalCounter &operator=(TradeIntervalCounter &&) = delete;

    ~TradeIntervalCounter();

    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
};
}  // namespace alphaone

#endif
