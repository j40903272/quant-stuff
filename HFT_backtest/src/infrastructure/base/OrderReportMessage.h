#ifndef _ORDER_REPORT_LISTENER_H
#define _ORDER_REPORT_LISTENER_H

#include "infrastructure/common/order/taifex/TMP.h"

enum class OrderReportSide
{
    Sell    = 0,
    Buy     = 1,
    Unknown = 2,
};

struct OrderReportMessageSent
{
    int             Price;
    int             Qty;
    OrderReportSide Side;
    char            OrderNo[TMP_ORDNO_LEN + 1];
    uint8_t         Type;
};

struct OrderReportMessageAccepted
{
    int             Price;
    int             Qty;
    OrderReportSide Side;
    char *          OrderNo;
};

struct OrderReportMessageRejected
{
    char *OrderNo;
};

struct OrderReportMessageCancelSent
{
    char OrderNo[TMP_ORDNO_LEN + 1];
};

struct OrderReportMessageCancelled
{
    char *OrderNo;
};

struct OrderReportMessageCancelFailed
{
    char *OrderNo;
};

struct OrderReportMessageExecuted
{
    int             Price;
    int             Qty;
    int             LeavesQty;
    OrderReportSide Side;
    char *          OrderNo;
};

struct OrderReportMessageModified
{
    int             Price;
    int             Qty;
    OrderReportSide Side;
    char *          OrderNo;
};

struct OrderReportMessageModifyFailed
{
    char *OrderNo;
};

struct OrderReportMessageInvalid
{
    char *      OrderNo;
    const char *Reason;
};

struct OrderReportMessageDropOrder
{
    char *OrderNo;
};

struct OrderReportMessageRejectByServer
{
    char *OrderNo;
    int   RejectReason;
    int   RestCount;
};

struct OrderReportMessageFastReport
{
    char *OrderNo;
    int   Price;
    int   MatchLots;
    int   TotalMatchLots;
};
#endif
