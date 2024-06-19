#ifndef _RISKSTATUS_H_
#define _RISKSTATUS_H_

#include <string>

namespace alphaone
{
enum class RiskStatus
{
    Invalid = 0,
    Good    = 1,

    /* *********************************************************************************************
    README (2020-05-25) [Andrew Kuo]:
    ├─   RiskStatus are supposed to be compared numerically in RiskController
    ├─   HardStop_TotalOrders should be the first HardStop
    └─   HardStop's should always be numerically larger than SoftStop's
    ********************************************************************************************* */

    SoftStop_OrdersPerSecond         = 2,
    SoftStop_OrdersPerMinute         = 3,
    SoftStop_TooFarFromLastTrade     = 4,
    SoftStop_TooFarCrossTouch        = 5,
    SoftStop_BouncesPerMinute        = 6,
    SoftStop_RepeatedOrdersPerMinute = 7,

    HardStop_TotalOrders = 10,
    HardStop_OrderPrice  = 11,

    HardStop_OrdersPerSecond         = 12,
    HardStop_OrdersPerMinute         = 13,
    HardStop_TooFarFromLastTrade     = 14,
    HardStop_TooFarCrossTouch        = 15,
    HardStop_BouncesPerMinute        = 16,
    HardStop_RepeatedOrdersPerMinute = 17,

    Exchange_RateLimits = 20,
};

std::string FromRiskStatusToString(const RiskStatus &status);
RiskStatus  FromStringToRiskStatus(const std::string &status);

enum class RiskControllerType
{
    Default,
    Gaia,
    Zeus,
    Vodka,
    Whisky,
};

std::string        FromRiskControllerTypeToString(const RiskControllerType &type);
RiskControllerType FromStringToRiskControllerType(const std::string &type);

}  // namespace alphaone

#endif
