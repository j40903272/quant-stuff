#include "MarketDataMessage.h"

namespace alphaone
{
const std::string MarketDataMessageTypeToString(MarketDataMessageType type)
{
    switch (type)
    {
    case MarketDataMessageType_Snapshot:
        return "SNS";
    case MarketDataMessageType_Add:
        return "ADD";
    case MarketDataMessageType_Delete:
        return "DEL";
    case MarketDataMessageType_ModifyWithPrice:
        return "MWP";
    case MarketDataMessageType_ModifyWithQty:
        return "MWQ";
    case MarketDataMessageType_Trade:
        return "TRD";
    case MarketDataMessageType_PacketEnd:
        return "PKE";
    default:
        return "INV";
    }
}

MarketDataMessage::MarketDataMessage(DataSourceType data_source_type)
    : type{data_source_type}, sequence_number{0}
{
}

void MarketDataMessage::Dump() const
{
    // clang-format off
    std::cout
        << "MarketDataMessage"
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
}  // namespace alphaone
