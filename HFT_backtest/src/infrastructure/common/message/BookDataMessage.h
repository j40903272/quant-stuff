#ifndef BOOKDATAMESSAGE_H
#define BOOKDATAMESSAGE_H

#include "infrastructure/common/message/MarketDataMessage.h"

namespace alphaone
{
class Order;

template <typename T>
struct DependentFalse
{
    static constexpr bool value = false;
};

struct BookDataMessage : protected MarketDataMessage
{
  public:
    BookDataMessage(const DataSourceType &data_source_type = DataSourceType::MarketByOrder);

    virtual ~BookDataMessage() = default;

    void ConstructBookMessageFromMarketMessage(const MarketDataMessage *mm);

    inline void SetMarketByOrderPrice(const BookPrice &price)
    {
        mbo.price = price;
    }

    inline void SetMarketByOrderQty(const BookQty &qty)
    {
        mbo.qty = qty;
    }

    inline void SetMarketByOrderSide(const BookSide &side)
    {
        mbo.side = side;
    }

    inline const Symbol *GetSymbol() const
    {
        return symbol;
    }

    inline Timestamp GetProviderTime() const
    {
        return provider_time;
    }

    inline Timestamp GetExchangeTime() const
    {
        return exchange_time;
    }

    inline int64_t GetSequenceNumber() const
    {
        return sequence_number;
    }

    inline BookPrice GetMarketByOrderPrice() const
    {
        return mbo.price;
    }

    inline BookQty GetMarketByOrderQty() const
    {
        return mbo.qty;
    }

    inline BookSide GetMarketByOrderSide() const
    {
        return mbo.side;
    }

    inline BookNord GetMarketByOrderNord() const
    {
        return mbo.nord;
    }

    inline MarketDataMessageType GetMarketDataMessageType() const
    {
        return market_data_message_type;
    }

    inline BookPrice GetMarketByPricePrice(const BookSide &side, const BookNolv &nolv) const
    {
        if (side == BID)
        {
            return mbp.bid_price[nolv] / symbol->GetDecimalConverter();
        }
        else
        {
            return mbp.ask_price[nolv] / symbol->GetDecimalConverter();
        }
    }

    inline BookQty GetMarketByPriceQty(const BookSide &side, const BookNolv &nolv) const
    {
        if (side == BID)
        {
            return mbp.bid_qty[nolv];
        }
        else
        {
            return mbp.ask_qty[nolv];
        }
    }

    inline BookPrice GetTradePrice() const
    {
        return trade.price / symbol->GetDecimalConverter();
    }

    inline BookQty GetTradeQty() const
    {
        return trade.qty;
    }

    inline BookSide GetTradeSide() const
    {
        return trade.side;
    }

    inline bool IsOverrunning() const
    {
        return is_overrunning;
    }

    inline bool IsRejoining() const
    {
        return is_rejoining;
    }

    virtual void Dump() const;

  protected:
    friend class Book;
    friend class MarketByOrderBook;
    friend class MarketByPriceBook;
    friend class OptionParityBook;
    friend class TWSEFileBook;

    bool is_overrunning;
    bool is_rejoining;
};

struct BookDataMessageAdd : public BookDataMessage
{
    bool   is_new_level_added_;  // if this add message has added a new level
    Order *order_;

    inline void SetOrder(Order *o)
    {
        order_ = o;
    }

    inline ExternalOrderId GetExternalOrderId() const
    {
        return mbo.order_id;
    }

    inline BookPrice GetTradePrice() const                                                 = delete;
    inline BookQty   GetTradeQty() const                                                   = delete;
    inline BookSide  GetTradeSide() const                                                  = delete;
    inline BookPrice GetMarketByPricePrice(const BookSide &side,
                                           const BookNolv &nolv) const                     = delete;
    inline BookQty   GetMarketByPriceQty(const BookSide &side, const BookNolv &nolv) const = delete;
    inline BookPrice GetMboPrice() const                                                   = delete;
};

struct BookDataMessageDelete : public BookDataMessage
{
    bool   is_level_fully_deleted_;  // if this del message has deleted an existing level
    Order *order_;

    inline void SetOrder(Order *o)
    {
        order_ = o;
    }

    inline ExternalOrderId GetExternalOrderId() const
    {
        return mbo.order_id;
    }

    inline bool IsTrade() const
    {
        return market_data_message_type == MarketDataMessageType_Trade;
    }

    inline BookPrice GetTradePrice() const                                                 = delete;
    inline BookQty   GetTradeQty() const                                                   = delete;
    inline BookSide  GetTradeSide() const                                                  = delete;
    inline BookPrice GetMarketByPricePrice(const BookSide &side,
                                           const BookNolv &nolv) const                     = delete;
    inline BookQty   GetMarketByPriceQty(const BookSide &side, const BookNolv &nolv) const = delete;
    inline BookPrice GetMboPrice() const                                                   = delete;
};

struct BookDataMessageModifyWithPrice : public BookDataMessage
{
    // I am not sure how we will handle ModifyWithPrice though. Will we treat ModifyWithPrice like
    // one deleting and one adding? Should we trace the pre-modification price and post-modification
    // price here?
    inline BookPrice GetTradePrice() const = delete;
    inline BookQty   GetTradeQty() const   = delete;
    inline BookSide  GetTradeSide() const  = delete;
    inline BookPrice GetMboPrice() const   = delete;
};

struct BookDataMessageModifyWithQty : public BookDataMessage
{
    // If the market information is not given in detail, then actually ModifyWithQty will be handled
    // in the same way as Delete. However, if the exchange really provide a by-order disclosure,
    // then ModifyWithQty contains different information than Delete. Orders being modified are
    // "more real" than orders being deleted, because the one who modified an order still wants part
    // of the order to be executed. Shoule we keep the pre-modification qty and post-modification
    // qty here?
    inline BookPrice GetTradePrice() const = delete;
    inline BookQty   GetTradeQty() const   = delete;
    inline BookSide  GetTradeSide() const  = delete;
    inline BookPrice GetMboPrice() const   = delete;
};

struct BookDataMessageTrade : public BookDataMessage
{
    ExternalOrderId counterparty_order_id_;

    void Dump() const override;

    inline ExternalOrderId GetExternalOrderId() const
    {
        return trade.order_id;
    }

    inline BookPrice GetMarketByOrderPrice() const                                         = delete;
    inline BookQty   GetMarketByOrderQty() const                                           = delete;
    inline BookSide  GetMarketByOrderSide() const                                          = delete;
    inline BookNord  GetMarketByOrderNord() const                                          = delete;
    inline BookPrice GetMarketByPricePrice(const BookSide &side,
                                           const BookNolv &nolv) const                     = delete;
    inline BookQty   GetMarketByPriceQty(const BookSide &side, const BookNolv &nolv) const = delete;
    inline BookPrice GetMboPrice() const                                                   = delete;
};

struct BookDataMessagePacketEnd : public BookDataMessage
{
    MarketDataMessageType last_market_data_type_;

    inline ExternalOrderId GetExternalOrderId() const
    {
        return last_market_data_type_ == MarketDataMessageType_Trade ? trade.order_id
                                                                     : mbo.order_id;
    }

    inline BookPrice GetTradePrice() const                                                 = delete;
    inline BookQty   GetTradeQty() const                                                   = delete;
    inline BookSide  GetTradeSide() const                                                  = delete;
    inline BookPrice GetMarketByOrderPrice() const                                         = delete;
    inline BookQty   GetMarketByOrderQty() const                                           = delete;
    inline BookSide  GetMarketByOrderSide() const                                          = delete;
    inline BookNord  GetMarketByOrderNord() const                                          = delete;
    inline BookPrice GetMarketByPricePrice(const BookSide &side,
                                           const BookNolv &nolv) const                     = delete;
    inline BookQty   GetMarketByPriceQty(const BookSide &side, const BookNolv &nolv) const = delete;
    inline BookPrice GetMboPrice() const                                                   = delete;
};

struct BookDataMessageSnapshot : public BookDataMessage
{
    inline BookPrice GetMarketByOrderPrice() const = delete;
    inline BookQty   GetMarketByOrderQty() const   = delete;
    inline BookSide  GetMarketByOrderSide() const  = delete;
    inline BookNord  GetMarketByOrderNord() const  = delete;
    inline BookPrice GetTradePrice() const         = delete;
    inline BookQty   GetTradeQty() const           = delete;
    inline BookSide  GetTradeSide() const          = delete;
    inline BookPrice GetMboPrice() const           = delete;
};

struct BookDataMessageSparseStop
{
    BookTrade     event_type;
    Timestamp     last_event_time_;
    Timestamp     event_loop_time_;
    const Symbol *symbol_;
};
}  // namespace alphaone

#endif
