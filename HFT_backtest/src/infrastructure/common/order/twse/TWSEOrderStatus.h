#ifndef _TWSEORDERSTATUS_H_
#define _TWSEORDERSTATUS_H_

#include <string>

namespace alphaone
{

enum class TWSEOrderStatus : int
{
    UNKNOWN                 = 0,
    SUCCESS                 = 1,
    INVENTORY_SHORTAGE      = 2,   // 庫存不足
    MARK_M                  = 3,   // 平盤以下不得借券賣出
    TID_NOTFOUND            = 4,   // 找不到TID
    PID_NOTFOUND            = 5,   // 找不到PID
    PID_NOUSE               = 6,   // PID不在可用清單
    BORROW_CANT_BUY         = 7,   // 借券不可買
    BLACKLIST               = 8,   // 不可交易清單
    WHITELIST               = 9,   // 可交易清單
    TID_ACCOUNT_NOT_MAPPING = 10,  // TID和Account不匹配
    ACCOUNT_NOT_FOUND       = 11,  // Account不存在
    NOT_SUPPORT_ODD_LOTS    = 12   // 不支援零股
};

std::string TWSEOrderStatusToString(const TWSEOrderStatus &s);


}  // namespace alphaone


#endif