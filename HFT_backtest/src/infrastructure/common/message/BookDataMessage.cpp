#include "BookDataMessage.h"

namespace alphaone
{
BookDataMessage::BookDataMessage(const DataSourceType &data_source_type)
    : MarketDataMessage{data_source_type}, is_overrunning{false}, is_rejoining{false}
{
}

void BookDataMessage::ConstructBookMessageFromMarketMessage(const MarketDataMessage *mm)
{
    symbol                   = mm->symbol;
    market_data_message_type = mm->market_data_message_type;
    provider_time            = mm->provider_time;
    exchange_time            = mm->exchange_time;
    sequence_number          = mm->sequence_number;
    mbo                      = mm->mbo;
    mbp                      = mm->mbp;
    trade                    = mm->trade;
    implied                  = mm->implied;
    mbo.price                = mm->GetMboPrice();
}

void BookDataMessage::Dump() const
{
    // clang-format off
    std::cout
        << "BookDataMessage"
        << " message_type=" << MarketDataMessageTypeToString(market_data_message_type)
        << " provider_time=" << provider_time
        << " exchange_time=" << exchange_time
        << " order_id=" << mbo.order_id
        << " price=" << mbo.price
        << " qty=" << mbo.qty
        << " side=" << (mbo.side == BID ? "BID" : "ASK")
        << '\n';
    // clang-format on
}

void BookDataMessageTrade::Dump() const
{
    // clang-format off
    std::cout
        << "BookDataMessage"
        << " message_type=" << MarketDataMessageTypeToString(market_data_message_type)
        << " provider_time=" << provider_time
        << " exchange_time=" << exchange_time
        << " order_id=" << trade.order_id
        << " price=" << GetTradePrice()
        << " qty=" << trade.qty
        << " side=" << (trade.side == BID ? "BID" : "ASK")
        << '\n';
    // clang-format on
}
}  // namespace alphaone
