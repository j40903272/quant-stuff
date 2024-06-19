#include "TWSEOrderStatus.h"

namespace alphaone
{

std::string TWSEOrderStatusToString(const TWSEOrderStatus &s)
{
    switch (s)
    {
    case TWSEOrderStatus::UNKNOWN:
        return "UNKNOWN";
    case TWSEOrderStatus::SUCCESS:
        return "SUCCESS";
    case TWSEOrderStatus::INVENTORY_SHORTAGE:
        return "INVENTORY_SHORTAGE";
    case TWSEOrderStatus::MARK_M:
        return "CANNOT_BORROW_SELL_UNDER_FLAT";
    case TWSEOrderStatus::TID_NOTFOUND:
        return "TID_NOTFOUND";
    case TWSEOrderStatus::PID_NOTFOUND:
        return "PID_NOTFOUND";
    case TWSEOrderStatus::PID_NOUSE:
        return "PID_NOUSE";
    case TWSEOrderStatus::BORROW_CANT_BUY:
        return "BORROW_CANT_BUY";
    case TWSEOrderStatus::BLACKLIST:
        return "BLACKLIST";
    case TWSEOrderStatus::WHITELIST:
        return "WHITELIST";
    case TWSEOrderStatus::TID_ACCOUNT_NOT_MAPPING:
        return "TID_ACCOUNT_NOT_MAPPING";
    case TWSEOrderStatus::ACCOUNT_NOT_FOUND:
        return "ACCOUNT_NOT_FOUND";
    case TWSEOrderStatus::NOT_SUPPORT_ODD_LOTS:
        return "NOT_SUPPORT_ODD_LOTS";
    default:
        return "DEFAULT";
    }
}


}  // namespace alphaone
