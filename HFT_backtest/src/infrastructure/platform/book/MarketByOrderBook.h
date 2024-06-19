#ifndef _MARKETBYORDERBOOK_H_
#define _MARKETBYORDERBOOK_H_

#include "infrastructure/base/Book.h"
#include "infrastructure/base/BookDataListener.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/side/Side.h"
#include "infrastructure/platform/book/LevelFactory.h"
#include "infrastructure/platform/book/OrderFactory.h"

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

namespace alphaone
{
/*
    MarketByOrderBook.{h, cpp} handles an orderbook updated in a by-order manner.
        ├─  Order is a limit order sent to the exchange by any market participant
        └─  Level is a certain price level. There is a list of orders with the same price in a level
*/
class MarketByOrderBook : public Book
{
  public:
    MarketByOrderBook(const Symbol *symbol, OrderFactory *order_factory,
                      LevelFactory *level_factory, const bool is_trading_through_delete,
                      const bool is_touching_on_delete, const bool is_queue_like,
                      const bool is_overrunning_checking, const bool is_rejoining_checking,
                      const bool is_emit_allowed_only_when_valid);
    MarketByOrderBook(const nlohmann::json &json, const Symbol *symbol, OrderFactory *order_factory,
                      LevelFactory *level_factory);
    MarketByOrderBook(const GlobalConfiguration *configuration, const Symbol *symbol,
                      OrderFactory *order_factory, LevelFactory *level_factory);
    MarketByOrderBook(const MarketByOrderBook &) = delete;
    MarketByOrderBook &operator=(const MarketByOrderBook &) = delete;

    ~MarketByOrderBook();

    // get book type -- need to be defined explicitly;
    DataSourceType GetType() const override
    {
        return DataSourceType::MarketByOrder;
    }

    // initialization
    void Done();

    // get methods for book information
    BookPrice GetPrice(const BookSide side, const BookNolv level = 0) const override;
    BookPrice GetPrice(BidType *, const BookNolv level = 0) const;
    BookPrice GetPrice(AskType *, const BookNolv level = 0) const;
    BookQty   GetQty(const BookSide side, const BookNolv level = 0) const override;
    BookQty   GetQty(BidType *, const BookNolv level = 0) const;
    BookQty   GetQty(AskType *, const BookNolv level = 0) const;
    BookNord  GetNord(const BookSide side, const BookNolv level = 0) const override;
    BookNord  GetNord(BidType *, const BookNolv level = 0) const;
    BookNord  GetNord(AskType *, const BookNolv level = 0) const;

    // for taifex, this method always returns 5. Nolv means number of levels.
    BookNolv GetNolv(const BookSide side) const override;

    // get methods for directional information -- utilized when using alpha models
    BookPrice GetMidPrice() const override;
    BookPrice GetWeightedPrice() const override;
    BookQty   GetMidQty() const override;
    BookQty   GetMarketOrderQty(const BookSide side) const;

    // get methods for gap information -- utilized when using gap models
    BookPrice GetSpread(const BookNolv bl = 0, const BookNolv al = 0) const override;
    BookPrice GetSpreadAsPercentOfMid(const BookNolv bl = 0, const BookNolv al = 0) const override;
    BookPrice GetSpreadAsPercentOfBid(const BookNolv bl = 0, const BookNolv al = 0) const override;
    BookPrice GetSpreadAsReturn(const BookNolv bl = 0, const BookNolv al = 0) const override;

    // get methods for complicated book information -- useful when designing market making strategy
    BookPrice GetPriceBehindQty(const BookSide &side, const BookQty &qty) const override;
    BookPrice GetPriceBeforeQty(const BookSide &side, const BookQty &qty) const override;
    BookQty   GetQtyBehindPrice(const BookSide &side, const BookPrice &price) const override;
    BookPrice GetPriceBehindLevel(const BookPrice &price, const BookSide &side,
                                  const BookNolv &nolv) const;

    // get methods
    Timestamp    GetTime(const BookSide side, const BookNolv nolv) const;
    BookNolv     GetNumberOfLevels(const BookSide side) const;
    const Level *GetTouchLevel(const BookSide side) const;
    const Level *GetTouchLevel(BidType *) const;
    const Level *GetTouchLevel(AskType *) const;
    const Level *GetWorstLevel(const BookSide side) const;
    const Order *GetOrder(const Level *level, const BookQty qty) const;
    const Level *GetLevelFromPrice(const BookSide &side, const BookPrice &price) const;
    const Level *GetLevelFromPrice(const BookSide &side, const BookPrice &price,
                                   BookNolv &nolv) const;

    // get methods for statistics
    Timestamp GetLastTradeTime() const override;
    BookPrice GetLastTradePrice() const override;
    BookQty   GetLastTradeQty() const override;
    BookSide  GetLastTradeSide() const;
    size_t    GetNumberUnCrossBooks() const;
    size_t    GetNumberUnlockBook() const;

    // public raw market data callbacks
    void OnAdd(const MarketDataMessage *) override;
    void OnDelete(const MarketDataMessage *) override;
    void OnModifyWithPrice(const MarketDataMessage *) override;
    void OnModifyWithQty(const MarketDataMessage *) override;
    void OnSnapshot(const MarketDataMessage *) override;
    void OnTrade(const MarketDataMessage *) override;
    void OnPacketEnd(const MarketDataMessage *) override;
    void OnSparseStop() override;

    // reset MarketByOrderBook instance if needed
    void Reset() override;
    void ResetOnlyBook();
    void ResetWithoutCounts();

    // dump book objects for debugging (or print on telnet directly if we would like to do this)
    void              Dump() const override;
    void              DumpPrettyBook(size_t window = 5) const;
    void              DumpSnapshot(size_t window = 5) const;
    const std::string DumpString(size_t window = 5) const override;
    const std::string DumpLevelOrder(BookSide side, BookNolv nolv) const;

    // utility function to parse direction of trades
    static BookSide ParseTradeSide(const BookPrice bid_price, const BookPrice ask_price,
                                   const BookPrice trade_price);

    // check
    bool IsValid() const override;
    bool IsValidBid() const override;
    bool IsValidAsk() const override;
    bool IsFlip() const override;
    bool IsFlipUp() const override;
    bool IsFlipDown() const override;

  protected:
    void EmitPreDelete() const override;
    void EmitPostAdd() const override;
    void EmitPostDelete() const override;

    // single order operation
    virtual void CreateOrder(Level *level, Order *order);
    virtual void CreateLevel(Level *level, Order *order);                   // insert to touch
    virtual void CreateLevel(Level *level, Order *order, Level *insert_l);  // insert to insert_l
    virtual void RemoveOrder(Level *level, Order *order, bool operated = false);
    virtual void DecreaseOrderQty(Level *level, Order *order, BookQty qty);
    virtual void IncreaseOrderQty(Level *level, Order *order, BookQty qty);

    // market data operation
    // removes certain qty (unspecified if one order or not); if qty is larger than current qty,
    // empties level but DOES NOT delete it
    void RemoveQty(Level *level, const BookQty qty, bool operated = false);
    // removes all orders; creates empty level
    void ClearOrders(Level *level);
    // deletes the actual level from all references and memory
    void RemoveLevel(Level *level);

    // we store bids and asks in reverse order so that we can use lower_bound to find the level
    // right before when we want to insert, and end() if its at the touch. HOWEVER reverse_iterator
    // seems about 2x slower than iterator, so we may want to change this in the future
    using map_l =
        __gnu_pbds::tree<BookPrice, Level *, std::less<BookPrice>, __gnu_pbds::rb_tree_tag,
                         __gnu_pbds::tree_order_statistics_node_update>;
    using map_g =
        __gnu_pbds::tree<BookPrice, Level *, std::greater<BookPrice>, __gnu_pbds::rb_tree_tag,
                         __gnu_pbds::tree_order_statistics_node_update>;
    map_l all_bids_;
    map_g all_asks_;

    Level *touch_bid_;
    Level *touch_ask_;

    BookPrice last_boundary_bid_;
    BookPrice last_boundary_ask_;

    struct MarketByPriceEntry
    {
        BookNolv  touch_index_[AskBid]{0, 0};
        BookNolv  worst_index_[AskBid]{0, 0};
        BookPrice touch_price_[AskBid]{INVALID_BID_PRICE, INVALID_ASK_PRICE};
        BookPrice worst_price_[AskBid]{INVALID_BID_PRICE, INVALID_ASK_PRICE};
    };

    const bool is_queue_like_;

    OrderFactory *order_factory_;
    LevelFactory *level_factory_;

    bool      is_trade_;
    Timestamp last_trade_time_;
    BookPrice last_trade_price_;
    BookQty   last_trade_qty_;
    BookSide  last_trade_side_;

    size_t book_statistic_uncross_book_total_;
    size_t book_statistic_unlock_book_total_;

    const bool is_trading_through_delete_;
    const bool is_touching_on_delete_;

    const bool is_overrunning_checking_;
    const bool is_rejoining_checking_;

    void ParseSnapshotOnBid(const MarketDataMessage *mm, const ExternalOrderId &order_id,
                            BookPrice price, BookQty qty, BookNord nord);
    void ParseSnapshotOnAsk(const MarketDataMessage *mm, const ExternalOrderId &order_id,
                            BookPrice price, BookQty qty, BookNord nord);

    MarketByPriceEntry market_by_price_entry_;
    void               ParseMarketByPriceEntry(const MarketDataMessage *mm);
    void               CleanLevelsBetweenMarketByPriceGap(const MarketDataMessage *mm);
    void               CleanLevelsNotAmongMarketByPrice(const MarketDataMessage *mm);

    virtual void UncrossMarketByOrderBook(const MarketDataMessage *mm);

    MarketDataMessage market_data_message_delete_;
    void CleanMarketByOrderBookAfterTrade(const BookSide side, const MarketDataMessage *mm);

    BookQty market_order_qty_[AskBid];
};

inline size_t MarketByOrderBook::GetNumberUnCrossBooks() const
{
    return book_statistic_uncross_book_total_;
}

inline size_t MarketByOrderBook::GetNumberUnlockBook() const
{
    return book_statistic_unlock_book_total_;
}
}  // namespace alphaone

#endif
