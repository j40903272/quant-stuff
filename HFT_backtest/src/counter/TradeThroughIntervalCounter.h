#ifndef _TRADETHROUGHINTERVALCOUNTER_H_
#define _TRADETHROUGHINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class TradeThroughIntervalCounter : public Counter
{
  public:
    TradeThroughIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    TradeThroughIntervalCounter(const TradeThroughIntervalCounter &) = delete;
    TradeThroughIntervalCounter &operator=(const TradeThroughIntervalCounter &) = delete;
    TradeThroughIntervalCounter(TradeThroughIntervalCounter &&)                 = delete;
    TradeThroughIntervalCounter &operator=(TradeThroughIntervalCounter &&) = delete;

    ~TradeThroughIntervalCounter();

    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
};
}  // namespace alphaone

#endif
