#include "RiskStatus.h"

namespace alphaone
{
std::string FromRiskStatusToString(const RiskStatus &status)
{
    switch (status)
    {
    case RiskStatus::Good:
        return "Good";
    case RiskStatus::SoftStop_OrdersPerSecond:
        return "SoftStop_OrdersPerSecond";
    case RiskStatus::SoftStop_OrdersPerMinute:
        return "SoftStop_OrdersPerMinute";
    case RiskStatus::SoftStop_TooFarFromLastTrade:
        return "SoftStop_TooFarFromLastTrade";
    case RiskStatus::SoftStop_TooFarCrossTouch:
        return "SoftStop_TooFarCrossTouch";
    case RiskStatus::SoftStop_BouncesPerMinute:
        return "SoftStop_BouncesPerMinute";
    case RiskStatus::SoftStop_RepeatedOrdersPerMinute:
        return "SoftStop_RepeatedOrdersPerMinute";
    case RiskStatus::HardStop_TotalOrders:
        return "HardStop_TotalOrders";
    case RiskStatus::HardStop_OrdersPerSecond:
        return "HardStop_OrdersPerSecond";
    case RiskStatus::HardStop_OrdersPerMinute:
        return "HardStop_OrdersPerMinute";
    case RiskStatus::HardStop_TooFarFromLastTrade:
        return "HardStop_TooFarFromLastTrade";
    case RiskStatus::HardStop_TooFarCrossTouch:
        return "HardStop_TooFarCrossTouch";
    case RiskStatus::HardStop_BouncesPerMinute:
        return "HardStop_BouncesPerMinute";
    case RiskStatus::HardStop_RepeatedOrdersPerMinute:
        return "HardStop_RepeatedOrdersPerMinute";
    case RiskStatus::Exchange_RateLimits:
        return "Exchange_RateLimits";
    default:
        return "Invalid";
    }
}

RiskStatus FromStringToRiskStatus(const std::string &status)
{
    if (status == "Good")
        return RiskStatus::Good;
    else if (status == "SoftStop_OrdersPerSecond")
        return RiskStatus::SoftStop_OrdersPerSecond;
    else if (status == "SoftStop_OrdersPerMinute")
        return RiskStatus::SoftStop_OrdersPerMinute;
    else if (status == "SoftStop_TooFarFromLastTrade")
        return RiskStatus::SoftStop_TooFarFromLastTrade;
    else if (status == "SoftStop_TooFarCrossTouch")
        return RiskStatus::SoftStop_TooFarCrossTouch;
    else if (status == "SoftStop_BouncesPerMinute")
        return RiskStatus::SoftStop_BouncesPerMinute;
    else if (status == "SoftStop_RepeatedOrdersPerMinute")
        return RiskStatus::SoftStop_RepeatedOrdersPerMinute;
    else if (status == "HardStop_TotalOrders")
        return RiskStatus::HardStop_TotalOrders;
    else if (status == "HardStop_OrdersPerSecond")
        return RiskStatus::HardStop_OrdersPerSecond;
    else if (status == "HardStop_OrdersPerMinute")
        return RiskStatus::HardStop_OrdersPerMinute;
    else if (status == "HardStop_TooFarFromLastTrade")
        return RiskStatus::HardStop_TooFarFromLastTrade;
    else if (status == "HardStop_TooFarCrossTouch")
        return RiskStatus::HardStop_TooFarCrossTouch;
    else if (status == "HardStop_BouncesPerMinute")
        return RiskStatus::HardStop_BouncesPerMinute;
    else if (status == "HardStop_RepeatedOrdersPerMinute")
        return RiskStatus::HardStop_RepeatedOrdersPerMinute;
    else if (status == "Exchange_RateLimits")
        return RiskStatus::Exchange_RateLimits;
    else
        return RiskStatus::Invalid;
}

std::string FromRiskControllerTypeToString(const RiskControllerType &type)
{
    switch (type)
    {
    case RiskControllerType::Gaia:
        return "Gaia";
    case RiskControllerType::Zeus:
        return "Zeus";
    case RiskControllerType::Vodka:
        return "Vodka";
    case RiskControllerType::Whisky:
        return "Whisky";
    default:
        return "Default";
    }
}

RiskControllerType FromStringToRiskControllerType(const std::string &type)
{
    if (type == "Gaia")
        return RiskControllerType::Gaia;
    else if (type == "Zeus")
        return RiskControllerType::Zeus;
    else if (type == "Vodka")
        return RiskControllerType::Vodka;
    else if (type == "Whisky")
        return RiskControllerType::Whisky;
    else
        return RiskControllerType::Default;
}
}  // namespace alphaone
