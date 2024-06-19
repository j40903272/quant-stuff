#ifndef _TAIFEXORDERSTATUS_H_
#define _TAIFEXORDERSTATUS_H_

#include <string>

namespace alphaone
{

// clang-format off【

enum class TaifexErrorStatus : unsigned char
{
    SUCCESS           = 0,
    TRADING_TIME_OVER = 1,  // 交易時間已結束
    NOT_START_RECEIVING_ORDER = 2,  // 尚未開始接收委託或暫停交易或者收單階段無此種委託
    NOT_START_RECEIVING_QUOTE = 3,   // 尚未開始接收報價
    PROCESSING                = 4,   // 該商品處理中，暫時不接受委託
    ORDER_NO_NOT_FOUND        = 5,   // 無此委託書編號
    IS_FULLY_FILLED           = 6,   // 委託或報價已完全成交
    ALREADY_CANCELLED         = 7,   // 原委託或報價已取消
    STOP_NEW_ORDER            = 8,   // 停單
    CLOSE_ORDER_ONLY          = 9,   // 限單
    ORDER_NO_LONGER_IN_BOOK   = 10,  // 此單已不在委託簿中,不得刪單、減量、改價
    INVALID_EXECTYPE          = 11,  // 執行類型(ExecType)錯誤
    INVALID_FUTURE_ID         = 12,  // 期貨商代號錯誤
    INVALID_BRANCH_ID         = 13,  // 分公司代號錯誤
    INVALID_ACCOUNT = 14,  // 交易人帳號錯誤或申報鉅額交易中之交易人帳號未開戶
    INVALID_INVESTER_NO      = 15,  // 投資人身份碼錯誤
    NO_CANCEL_MODIFY_PERIOD  = 16,  // 不可刪改單期間，退回刪改單
    DUPLICATE_ORDER_NO       = 17,  // 委託書或報價單編號重複
    INVALID_ORDER_NO         = 18,  // 委託書或報價單編號錯誤
    NO_SAME_SIDE_LIMIT_ORDER = 19,  // 無同方限價委託
    INVALID_PID              = 20,  // 商品代號錯誤
    PRICE_OVER_LIMIT         = 21,  // 超過該商品漲跌停價
    INVALID_QTY              = 22,  // 委託數量錯誤
    INVALID_SIDE             = 24,  // 買賣別錯誤
    INVALID_QRDTYPE          = 25,  // 委託種類錯誤(OrdType)
    ORDTYPE_PRICE_NOT_MATCH  = 26,  // 委託種類與價位不符
    INVALID_TIMEINFORCE      = 27,  // 委託條件錯誤(TimeInForce)
    INVALID_POSITIONEFFECT   = 28,  // 開平倉碼錯誤(PositionEffect)
    INVALID_PRICE_TICK       = 29,  // 價格與基本跳動價位不符
    EXPIRED_PRODUCT          = 30,  // 商品已過期
    FUTURE_ID_TYPE_ACNO_MISS = 31,  // 期貨商代號、報價商品類別、造市者帳號不一致
    MODIFY_QTY_LARGER        = 32,  // 警告減量成功,但變更量大於原委託剩餘可減量
    OVER_QUOTAION_SPREAD     = 33,  // 買賣價不得超過該報價商品之價差限制
    BID_OVER_ASK             = 34,  // 買價超過賣價
    QUOTE_MODIFY_QTY_ZERO = 35,  // 報價單邊改價，未改價邊剩餘口數0，不接受單邊改價
    QUOTE_QTY_BELOW_LIMIT = 36,  // 單邊報價剩餘量少於最低數量限制
    MARKET_ORDER_NOT_ALLOWED = 38,  // 市價單(及一定範圍市價委託)不允許當日有效委託
    QUOTE_MODIFY_PRICE_WRONG = 40,  // 報價單邊改價，價格輸入錯誤
    PRICE_OVER_DYNAMIC_RANGE =
        47,  // 該筆委託可能成交價或未成交委託價超過動態價格穩定區間上限或下限，剩餘口數已刪除
    PRICE_OVER_DYNAMIC_LIMIT =
        48,  // Price 或 leg_px 為該筆委託因動態價格穩定措施退單時之上限或下限價格
    FOK_NOT_FILLED           = 51,   // FOK 單未成交,系統已刪除
    IOC_PARTIALLY_FILLED     = 52,   // IOC 單已部份成交
    IOC_NOT_FILLED           = 53,   // IOC 單未成交,系統已刪除
    CANCEL_REDUCE_SIDE_WRONG = 61,   // 刪單減量時原買賣別不符
    CANCEL_REDUCE_PROD_WRONG = 62,   // 刪單減量時原商品別不符
    IOC_CANNOT_MODIFY        = 63,   // IOC 委託不允許改價
    KILL_SWITCH_ON           = 65,   // 啟動 Kill Switch 作業，暫不接受此種委託
    INVALID_CONTRACT         = 77,   // 無此商品契約
    SESSION_OVER_PVC_LIMIT   = 240,  // 該 session 超過設定之流量值
    SESSION_STOP_DUE_TO_OVER =
        241,  // 該 session 於連續時間內都超過設定之流量值，本公司將自動暫停收單一段時間
    SESSION_PVC_OVER_80 = 248,  // 警告該 session 以達設定之流量值80%
    SESSION_PVC_OVER_90 = 249,  // 警告該 session 以達設定之流量值90%
};

// clang-format on

std::string TaifexErrorStatusToString(const TaifexErrorStatus &status);

}  // namespace alphaone


#endif