#ifndef _MARKETBYPRICEBOOK_H_
#define _MARKETBYPRICEBOOK_H_

#include "infrastructure/base/Book.h"
#include "infrastructure/base/BookDataListener.h"

namespace alphaone
{
class MarketByPriceBook : public Book
{
  public:
    MarketByPriceBook(const Symbol *symbol, const bool is_emit_allowed_only_when_valid = false);
    ~MarketByPriceBook();

    // get book type -- need to be defined explicitly;
    DataSourceType GetType() const final
    {
        return DataSourceType::MarketByPrice;
    }

    // get methods for book information
    BookPrice GetPrice(const BookSide side, const BookNolv nolv = 0) const final;
    BookQty   GetQty(const BookSide side, const BookNolv nolv = 0) const final;
    BookNord  GetNord(const BookSide side, const BookNolv nolv = 0) const final;

    // for taifex, this method always returns 5. Nolv means number of levels.
    BookNolv GetNolv(const BookSide side) const final;

    // get methods for directional information -- utilized when using alpha models
    BookPrice GetMidPrice() const final;
    BookPrice GetWeightedPrice() const final;
    BookQty   GetMidQty() const final;

    // get methods for gap information -- utilized when using gap models
    BookPrice GetSpread(const BookNolv b_lvl = 0, const BookNolv a_lvl = 0) const final;
    BookPrice GetSpreadAsPercentOfMid(const BookNolv b_lvl = 0,
                                      const BookNolv a_lvl = 0) const final;
    BookPrice GetSpreadAsPercentOfBid(const BookNolv b_lvl = 0,
                                      const BookNolv a_lvl = 0) const final;
    BookPrice GetSpreadAsReturn(const BookNolv b_lvl = 0, const BookNolv a_lvl = 0) const final;

    // get methods for complicated book information -- useful when designing market making strategy
    BookPrice GetPriceBehindQty(const BookSide &side, const BookQty &qty) const final;
    BookPrice GetPriceBeforeQty(const BookSide &side, const BookQty &qty) const final;
    BookQty   GetQtyBehindPrice(const BookSide &side, const BookPrice &price) const final;

    // get methods for statistics
    Timestamp GetLastTradeTime() const final;
    BookPrice GetLastTradePrice() const final;
    BookQty   GetLastTradeQty() const final;

    // public raw market data callbacks
    void OnAdd(const MarketDataMessage *) final;              // self
    void OnDelete(const MarketDataMessage *) final;           // self
    void OnModifyWithPrice(const MarketDataMessage *) final;  // self
    void OnModifyWithQty(const MarketDataMessage *) final;    // self
    void OnSnapshot(const MarketDataMessage *) final;         // self
    void OnTrade(const MarketDataMessage *) final;            // self
    void OnPacketEnd(const MarketDataMessage *) final;        // self
    void OnSparseStop() final;                                // self
  

    // reset MarketByOrderBook instance if needed
    void Reset() final;

    // dump book objects for debugging (or print on telnet directly if we would like to do this)
    void              Dump() const final;
    const std::string DumpString(size_t window = 5) const final;

    // check
    bool IsValid() const final;
    bool IsValidBid() const final;
    bool IsValidAsk() const final;
    bool IsFlip() const final;
    bool IsFlipUp() const final;
    bool IsFlipDown() const final;

  private:
    Timestamp               market_data_message_market_by_price_time_;
    MarketDataMessage_MBP   market_data_message_market_by_price_;
    Timestamp               market_data_message_trade_time_;
    MarketDataMessage_Trade market_data_message_trade_;
};
}  // namespace alphaone
#endif
