#ifndef _FIXDEFINE_H_
#define _FIXDEFINE_H_

enum class FIX_TAG : int
{
    Account               = 1,
    AvgPx                 = 6,
    ClOrdID               = 11,
    CumQty                = 14,
    ExecID                = 17,
    LastPx                = 31,
    LastQty               = 32,
    MsgSeqNum             = 34,
    MsgType               = 35,
    NewSeqNo              = 36,
    OrderID               = 37,
    OrderQty              = 38,
    OrdStatus             = 39,
    OrdType               = 40,
    OrigClOrdID           = 41,
    Price                 = 44,
    SenderCompID          = 49,
    SenderSubID           = 50,
    SendingTime           = 52,
    Side                  = 54,
    Symbol                = 55,
    TargetCompID          = 56,
    Text                  = 58,
    TimeInForce           = 59,
    TransactTime          = 60,
    OrdRejReason          = 103,
    TestReqID             = 112,
    ExecType              = 150,
    LeavesQty             = 151,
    ExecRestatementReason = 378,
    TwseIvacnoFlag        = 10000,
    TwseOrdType           = 10001,
    TwseExCode            = 10002,
};

#endif