#include "infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{
const std::string FromProductTypeToString(ProductType type)
{
    switch (type)
    {
    case ProductType::Future:
        return "FUTURE";
    case ProductType::Option:
        return "OPTION";
    case ProductType::Security:
        return "SECURITY";
    case ProductType::Perp:
        return "PERP";
    default:
        return "None";
    }
}

ProductType FromStringToProductType(const std::string type)
{
    if (type == "Future" or type == "FUTURE")
    {
        return ProductType::Future;
    }
    else if (type == "Option" or type == "OPTION")
    {
        return ProductType::Option;
    }
    else if (type == "Security" or type == "SECURITY")
    {
        return ProductType::Security;
    }
    else if (type == "Perp" or type == "PERP")
    {
        return ProductType::Perp;
    }
    else
    {
        return ProductType::None;
    }
}

const std::string FromDataSourceTypeToString(DataSourceType type)
{
    switch (type)
    {
    case DataSourceType::MarketByPrice:
        return "MarketByPrice";
    case DataSourceType::MarketByOrder:
        return "MarketByOrder";
    default:
        return "Invalid";
    }
}

DataSourceType FromStringToDataSourceType(const std::string type)
{
    if (type == "MarketByPrice")
    {
        return DataSourceType::MarketByPrice;
    }
    else if (type == "MarketByOrder")
    {
        return DataSourceType::MarketByOrder;
    }
    else
    {
        return DataSourceType::Invalid;
    }
}
}  // namespace alphaone
