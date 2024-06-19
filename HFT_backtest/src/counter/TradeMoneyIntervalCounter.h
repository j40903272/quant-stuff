#ifndef _TRADEMONEYINTERVALCOUNTER_H_
#define _TRADEMONEYINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"
#include "infrastructure/platform/manager/ObjectManager.h"

namespace alphaone
{
class TradeMoneyIntervalCounter : public Counter
{
  public:
    TradeMoneyIntervalCounter(const ObjectManager *object_manager,
                              MultiBookManager *multi_book_manager, const nlohmann::json &spec);
    TradeMoneyIntervalCounter(const TradeMoneyIntervalCounter &) = delete;
    TradeMoneyIntervalCounter &operator=(const TradeMoneyIntervalCounter &) = delete;
    TradeMoneyIntervalCounter(TradeMoneyIntervalCounter &&)                 = delete;
    TradeMoneyIntervalCounter &operator=(TradeMoneyIntervalCounter &&) = delete;

    ~TradeMoneyIntervalCounter();

    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) final;
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) final;

    BookQty GetTotalTradeQty() const;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    double ParseThreshold(const std::vector<Ohlcv<BookPrice, BookQty>> &ohlcvs) const;

    const Symbol *symbol_;
    const double  accuracy_;
    const double  threshold_;
    BookQty       packet_trade_money_;
    BookQty       this_trade_money_;
    BookQty       last_trade_money_;
};
}  // namespace alphaone

#endif
