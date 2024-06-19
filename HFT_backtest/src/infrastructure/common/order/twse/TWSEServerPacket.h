#ifndef _TWSETRADESERVERPACKET_H_
#define _TWSETRADESERVERPACKET_H_

#include "TWSEFixFormat.h"
namespace alphaone
{
#pragma pack(1)
enum class TWSETRADESERVER_ACTION : int
{
    UNKNOWN          = 0,
    LOGIN_REQUEST    = 1,
    LOGIN_REPLY      = 2,
    ORDER_NEW        = 3,
    ORDER_CANCEL     = 4,
    REPORT           = 5,
    DROPORDER        = 6,
    REJECT_NEW_ORDER = 7,
};

enum class TWSETRADESERVER_STATUS : int
{
    UNKNOWN       = 0,
    LOGIN_SUCCESS = 1,
    LOGIN_FAIL    = 2,
};

struct TWSETradeServerPacket_Header
{
    int Action;
};

struct TWSETradeServerPacket_LoginRequest
{
    TWSETradeServerPacket_Header Header;
    int                          Version;
    char                         OrderID[2];
    char                         KillOrderOnDisconnect;
};

struct TWSETradeServerPacket_LoginReply
{
    TWSETradeServerPacket_Header Header;
    int                          Status;
    char                         OrderNo[5];
    char                         ClOrdID[4];
};

struct TWSETradeServerPacket_TWSEReport
{
    TWSETradeServerPacket_Header Header;
    char                         Data[1024];
};

struct TWSETradeServerPacket_DropOrder
{
    TWSETradeServerPacket_Header Header;
    TWSEFixNewOrder              NewOrder;
};

struct TWSETradeServerPacket_NewOrder
{
    TWSETradeServerPacket_Header Header;
    char                         ClOrdID[12];  // 7798J1abcdA7
    char                         OrderNo[5];
    char                         Account[7];
    char                         Symbol[6];
    char                         Side[1];
    char                         OrderQty[3];
    char                         TimeInForce[1];
    char                         Price[10];
    char                         TwseOrdType[1];
    char                         OrdType[1];
    char                         TWSEExCode[1];
    char                         TradeSession[1];
    int                          SessionIndex;
};

struct TWSETradeServerPacket_CancelOrder
{
    TWSETradeServerPacket_Header Header;
    char                         OrigClOrdID[12];
    char                         ClOrdID[12];
    char                         OrderNo[5];
    char                         Account[7];
    char                         Symbol[6];
    char                         Side[1];
    char                         TradeSession[1];
    int                          SessionIndex;
};

struct TWSETradeServerPacket_RejectNewOrder
{
    TWSETradeServerPacket_Header Header;
    char                         ClOrdID[12];  // 7798J1abcdA7
    char                         OrderNo[5];
    char                         Account[7];
    char                         Symbol[6];
    char                         Side[1];
    char                         OrderQty[3];
    char                         TimeInForce[1];
    char                         Price[10];
    char                         TwseOrdType[1];
    char                         OrdType[1];
    char                         TWSEExCode[1];
    char                         TradeSession[1];
    int                          RejectReason;
    int                          RestCount;
};

#pragma pack()
}  // namespace alphaone
#endif