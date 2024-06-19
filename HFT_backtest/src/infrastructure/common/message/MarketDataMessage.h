#ifndef _MARKETDATAMESSAGE_H_
#define _MARKETDATAMESSAGE_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <cstdio>
#include <cstring>

namespace alphaone
{
enum MarketDataMessageType : uint8_t
{
    MarketDataMessageType_Invalid         = 0,
    MarketDataMessageType_Snapshot        = 1,
    MarketDataMessageType_Add             = 2,
    MarketDataMessageType_Delete          = 3,
    MarketDataMessageType_ModifyWithPrice = 4,
    MarketDataMessageType_ModifyWithQty   = 5,
    MarketDataMessageType_Trade           = 6,
    MarketDataMessageType_PacketEnd       = 7,
    MarketDataMessageType_Implied         = 8,
    MarketDataMessageType_TPrice          = 9,
};

const std::string MarketDataMessageTypeToString(MarketDataMessageType type);

struct MarketDataMessage_MBP
{
    MarketDataMessage_MBP &operator=(const MarketDataMessage_MBP &mbp)
    {
        memcpy(this, &mbp, sizeof(mbp));
        return *this;
    }

    BookPrice bid_price[10];
    BookPrice ask_price[10];
    BookQty   bid_qty[10];
    BookQty   ask_qty[10];
    uint8_t   count;
    bool      is_packet_end;
};

struct MarketDataMessage_MBO
{
    ExternalOrderId order_id;
    BookPrice       price;
    BookQty         qty;
    BookNord        nord;
    BookSide        side;
    bool            is_packet_end;
};

struct MarketDataMessage_Trade
{
    MarketDataMessage_Trade &operator=(const MarketDataMessage_Trade &trade)
    {
        memcpy(this, &trade, sizeof(trade));
        return *this;
    }

    ExternalOrderId order_id;
    ExternalOrderId counterparty_order_id;
    BookPrice       price;
    BookQty         qty;
    BookSide        side;
    bool            is_packet_end;
    // HACK: for TWSE/OTC
    bool is_not_duplicate_{true};
};

struct MarketDataMessage_Implied
{
    BookPrice bid_price[5];
    BookPrice ask_price[5];
    BookQty   bid_qty[5];
    BookQty   ask_qty[5];
    uint8_t   count;
    bool      is_packet_end;
};

struct MarketDataMessage_TPrice
{
    int  Price;
    int  BSCode;
    int  Type;
    bool is_packet_end;
};

struct MarketDataMessage
{
    MarketDataMessage(DataSourceType data_source_type);

    inline BookPrice GetMbpBidPrice(BookNolv l) const
    {
        return mbp.bid_price[l] / symbol->GetDecimalConverter();
    }

    inline BookPrice GetMbpAskPrice(BookNolv l) const
    {
        return mbp.ask_price[l] / symbol->GetDecimalConverter();
    }

    inline BookPrice GetMbpBidQty(BookNolv l) const
    {
        return mbp.bid_qty[l];
    }

    inline BookPrice GetMbpAskQty(BookNolv l) const
    {
        return mbp.ask_qty[l];
    }

    inline BookPrice GetTradePrice() const
    {
        return trade.price / symbol->GetDecimalConverter();
    }

    inline BookQty GetTradeQty() const
    {
        return trade.qty;
    }

    inline BookPrice GetMboPrice() const
    {
        return symbol ? mbo.price / symbol->GetDecimalConverter() : mbo.price;
    }

    void Dump() const;

    DataSourceType            type;
    const Symbol *            symbol;
    MarketDataMessageType     market_data_message_type;
    Timestamp                 provider_time;
    Timestamp                 exchange_time;
    int64_t                   sequence_number;
    MarketDataMessage_MBP     mbp;
    MarketDataMessage_MBO     mbo;
    MarketDataMessage_Trade   trade;
    MarketDataMessage_Implied implied;
    MarketDataMessage_TPrice  tprice;
    ProviderID                provider_id;
};

}  // namespace alphaone

#endif
