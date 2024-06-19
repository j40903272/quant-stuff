#ifndef _TRADEQTYINTERVALCOUNTER_H_
#define _TRADEQTYINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class TradeQtyIntervalCounter : public Counter
{
  public:
    TradeQtyIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                            const nlohmann::json &spec);
    TradeQtyIntervalCounter(const TradeQtyIntervalCounter &) = delete;
    TradeQtyIntervalCounter &operator=(const TradeQtyIntervalCounter &) = delete;
    TradeQtyIntervalCounter(TradeQtyIntervalCounter &&)                 = delete;
    TradeQtyIntervalCounter &operator=(TradeQtyIntervalCounter &&) = delete;

    ~TradeQtyIntervalCounter();

    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) override;

    BookQty GetTotalTradeQty() const;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    const BookQty threshold_;
    BookQty       total_trade_qty_;
    BookQty       last_total_trade_qty_;
};
}  // namespace alphaone

#endif
