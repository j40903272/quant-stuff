#ifndef ORDERREPORTSOURCE_H
#define ORDERREPORTSOURCE_H

#include "infrastructure/base/OrderReportMessage.h"
#include "infrastructure/base/TaifexOrderReportListener.h"
#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/platform/manager/ObjectManager.h"

#include <string.h>
#include <tuple>
#include <vector>

namespace alphaone
{

template <class T>
class OrderReportSource
{
  public:
    OrderReportSource(ObjectManager *object_manager)
    {
        object_manager_ = object_manager;

        memset(&order_report_accepted_, 0, sizeof(order_report_accepted_));
        memset(&order_report_rejected_, 0, sizeof(order_report_rejected_));
        memset(&order_report_cancelled_, 0, sizeof(order_report_cancelled_));
        memset(&order_report_cancel_failed_, 0, sizeof(order_report_cancel_failed_));
        memset(&order_report_executed_, 0, sizeof(order_report_executed_));
        memset(&order_report_modified_, 0, sizeof(order_report_modified_));
        memset(&order_report_modify_failed_, 0, sizeof(order_report_modify_failed_));
        memset(&order_report_drop_order_, 0, sizeof(order_report_drop_order_));
        memset(&order_report_reject_by_server_, 0, sizeof(order_report_reject_by_server_));
        memset(&order_report_fast_report_, 0, sizeof(order_report_fast_report_));

        for (auto item : object_manager->GetGlobalConfiguration()->GetJson().at("Universe"))
        {
            const Symbol *symbol{
                object_manager->GetSymbolManager()->GetSymbolByString(item.get<std::string>())};
            symbol_list_.push_back(symbol);
        }
        // Due to we use binary search thus we need to sort after inserting all the listening
        // symbols
        SortOrderReportSource();
    }

    OrderReportSource(const OrderReportSource &) = delete;
    OrderReportSource &operator=(const OrderReportSource &) = delete;

    virtual ~OrderReportSource() = default;

    virtual void SortOrderReportSource()
    {
        std::sort(symbol_list_.begin(), symbol_list_.end(),
                  [](const Symbol *p1, const Symbol *p2) -> bool
                  {
                      const std::string &pid1 = p1->GetDataSourcePid();
                      const std::string &pid2 = p2->GetDataSourcePid();

                      int ret = memcmp(pid1.c_str(), pid2.c_str(), pid1.size());

                      return ret < 0;
                  });
    }

    virtual void AddOrderReportListener(T *listener)
    {
        if (std::find(order_report_listener_.begin(), order_report_listener_.end(), listener) ==
            order_report_listener_.end())
        {
            order_report_listener_.push_back(listener);
        }
    }

    // give control to OrderReportSource, and OrderReportSource will callback OrderReportListener
    virtual void Process(const Timestamp &event_loop_time) = 0;

    // for simulation
    virtual const Timestamp PeekTimestamp()
    {
        return Timestamp::now();
    }

  protected:
    const Symbol *GetSymbol(const char *pid, const int len)
    {
        long int head{0};
        long int tail{static_cast<long int>(symbol_list_.size() - 1)};

        while (head <= tail)
        {
            long int mid{(head + tail) / 2};

            int ret{memcmp(pid, symbol_list_[mid]->GetDataSourcePid().c_str(), len)};

            if (ret > 0)
            {
                head = mid + 1;
            }
            else if (ret < 0)
            {
                tail = mid - 1;
            }
            else
            {
                return symbol_list_[mid];
            }
        }
        return nullptr;
    }

    std::vector<T *> &GetOrderReportListener()
    {
        return order_report_listener_;
    }

  protected:
    OrderReportMessageAccepted       order_report_accepted_;
    OrderReportMessageRejected       order_report_rejected_;
    OrderReportMessageCancelled      order_report_cancelled_;
    OrderReportMessageCancelFailed   order_report_cancel_failed_;
    OrderReportMessageExecuted       order_report_executed_;
    OrderReportMessageModified       order_report_modified_;
    OrderReportMessageModifyFailed   order_report_modify_failed_;
    OrderReportMessageDropOrder      order_report_drop_order_;
    OrderReportMessageRejectByServer order_report_reject_by_server_;
    OrderReportMessageFastReport     order_report_fast_report_;

  private:
    std::vector<T *>            order_report_listener_;
    std::vector<const Symbol *> symbol_list_;
    ObjectManager *             object_manager_;
};
}  // namespace alphaone

#endif
