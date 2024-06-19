#include "infrastructure/common/util/EnumToString.h"

#define UNDEFINE_STRING "Undefined"

namespace alphaone
{
std::string EnumToString::ToString(const BookTrade &book_trade)
{
    switch (book_trade)
    {
    case BookTrade::BookPart:
        return "Book";
    case BookTrade::TradePart:
        return "Trade";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const Side &side)
{
    switch (side)
    {
    case Side::Ask:
        return "Ask";
    case Side::Bid:
        return "Bid";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const ProductType &product_type)
{
    switch (product_type)
    {
    case ProductType::Future:
        return "FUTURE";
    case ProductType::Option:
        return "OPTION";
    case ProductType::Security:
        return "SECURITY";
    case ProductType::Warrant:
        return "WARRANT";
    case ProductType::Perp:
        return "PERP";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const ENUM_CPFlag &cpflag)
{
    switch (cpflag)
    {
    case ENUM_CPFlag::CALL:
        return "Call";
    case ENUM_CPFlag::PUT:
        return "Put";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const DataSourceType &data_source_type)
{
    switch (data_source_type)
    {
    case DataSourceType::MarketByPrice:
        return "MarketByPrice";
    case DataSourceType::MarketByOrder:
        return "MarketByOrder";
    case DataSourceType::OptionParity:
        return "OptionParity";
    case DataSourceType::TPrice:
        return "TPrice";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const DataSourceID &data_source_id)
{
    switch (data_source_id)
    {
    case DataSourceID::TAIFEX_FUTURE:
        return "Taifex_Future";
    case DataSourceID::TAIFEX_OPTION:
        return "Taifex_Option";
    case DataSourceID::TSE:
        return "Taifex_TSE";
    case DataSourceID::OTC:
        return "Taifex_OTC";
    case DataSourceID::SGX_FUTURE:
        return "SGX";
    case DataSourceID::CME:
        return "CME";
    case DataSourceID::TPRICE:
        return "TPRICE";
    case DataSourceID::SGX_REALTIME_FUTURE:
        return "SGX_Realtime_Future";
    case DataSourceID::TWSE_DATA_FILE:
        return "TWSE_DATA_FILE";
    case DataSourceID::BINANCE_PERP:
        return "BINANCE_PERP";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const ProviderID &provider_id)
{
    switch (provider_id)
    {
    case ProviderID::TAIFEX_FUTURE:
        return "Taifex_Future";
    case ProviderID::TAIFEX_OPTION:
        return "Taifex_Option";
    case ProviderID::TSE:
        return "Taifex_TSE";
    case ProviderID::OTC:
        return "Taifex_OTC";
    case ProviderID::SGX_FUTURE:
        return "SGX_Future";
    case ProviderID::CME:
        return "CME";
    case ProviderID::TPRICE:
        return "TPRICE";
    case ProviderID::SGX_REALTIME_FUTURE:
        return "SGX_Realtime_Future";
    case ProviderID::TWSE_DATA_FILE:
        return "TWSE_DATA_FILE";
    case ProviderID::AlphaOne:
        return "AlphaOne";
    case ProviderID::BINANCE_PERP:
        return "BINANCE_PERP";
    default:
        return UNDEFINE_STRING;
    }
}

std::string EnumToString::ToString(const EngineEventLoopType &event_loop_type)
{
    switch (event_loop_type)
    {
    case EngineEventLoopType::Simulation:
        return "Simulation";
    case EngineEventLoopType::Production:
        return "Production";
    default:
        return UNDEFINE_STRING;
    }
}

}  // namespace alphaone
