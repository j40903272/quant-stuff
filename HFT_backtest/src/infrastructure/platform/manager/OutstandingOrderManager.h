#ifndef _OUTSTANDINGORDERMANAGER_H_
#define _OUTSTANDINGORDERMANAGER_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <map>

namespace alphaone
{

struct OutstandingOrder
{
  public:
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const OutstandingOrder &o)
    {
        os << "[" << o.price_ << "," << o.sent_qty_ << "," << o.leaves_qty_ << ","
           << o.expired_time_ << "]";
        return os;
    }
    BookPrice    price_;
    BookQty      sent_qty_;
    BookQty      leaves_qty_;
    Timestamp    expired_time_;
    Timestamp    sent_time_;
    Timestamp    last_cancel_time_;
};

struct CancelOrderInfo
{
  public:
    template <typename OStream>
    friend OStream &operator<<(OStream &os, const CancelOrderInfo &co)
    {
        os << "[" << co.side_ << "," << co.price_ << "," << co.last_cancel_time_ << "]";
        return os;
    }
    BookSide     side_;
    BookPrice    price_;
    Timestamp    last_cancel_time_;
};

using OrderMap      = std::map<OrderNo, OutstandingOrder>;
using PriceOrderMap = std::map<BookPrice, std::pair<BookQty, OrderMap>, dynamic_compare>;
using FlatOrderMap  = std::unordered_map<OrderNo, CancelOrderInfo>;

class OutstandingOrderManager
{

  public:
    OutstandingOrderManager();
    ~OutstandingOrderManager();
    // Insert new Outstanding order
    bool InsertOrder(const Timestamp &event_loop_time, BookSide side, OrderNo orderno,
                     BookPrice price, BookQty qty, const Timestamp &expired_time);
    bool InsertCancelOrder(BookSide side, OrderNo orderno, BookPrice price);
    // Remove Outstanding order
    bool                  RemoveOrder(OrderNo orderno);
    std::pair<bool, bool> RemoveCancelOrder(OrderNo orderno, bool is_clean_outstanding = true);
    // Update Outstanding order
    bool UpdateOrder(OrderNo orderno, BookQty qty, bool is_change_qty = false);
    // iterate all orders to check expired or not
    void CleanExpiredOrders(const Timestamp &event_loop_time);
    // print all outstanding orders by price
    void PrintOrders();

    BookQty GetQtyFromPrice(BookSide side, BookPrice price);
    BookQty GetSharesPotential(BookSide side);

    OrderMap *     GetOrderMapFromPrice(BookSide side, BookPrice price);
    PriceOrderMap *GetPriceOrderMapFromSide(BookSide side);
    FlatOrderMap * GetOrderMap();
    FlatOrderMap * GetCancellingMap();

  private:
    PriceOrderMap price_order_map_[AskBid];
    FlatOrderMap  orderno_map_;
    std::string   side_string_[AskBid];
    BookQty       share_potential_[AskBid];
    FlatOrderMap  orderno_to_be_cancel_;
};


}  // namespace alphaone


#endif