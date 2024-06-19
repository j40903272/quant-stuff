#ifndef _TAIFEX_ORDER_REPORT_LISTENER_H
#define _TAIFEX_ORDER_REPORT_LISTENER_H

#include "infrastructure/base/OrderReportMessage.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/order/taifex/TMP.h"
#include "infrastructure/common/symbol/Symbol.h"

namespace alphaone
{
class TaifexOrderReportListener
{
  public:
    virtual void OnAccepted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageAccepted *o, void *packet) = 0;

    virtual void OnRejected(const Timestamp event_loop_time, OrderReportMessageRejected *o,
                            void *packet) = 0;

    virtual void OnCancelled(const Timestamp event_loop_time, const Symbol *symbol,
                             OrderReportMessageCancelled *o, void *packet) = 0;

    virtual void OnCancelFailed(const Timestamp event_loop_time, OrderReportMessageCancelFailed *o,
                                void *packet) = 0;

    virtual void OnExecuted(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageExecuted *o, void *packet) = 0;

    virtual void OnModified(const Timestamp event_loop_time, const Symbol *symbol,
                            OrderReportMessageModified *o, void *packet) = 0;

    virtual void OnModifyFailed(const Timestamp event_loop_time, OrderReportMessageModifyFailed *o,
                                void *packet) = 0;
};
}  // namespace alphaone
#endif
