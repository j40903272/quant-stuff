#include "OutstandingOrderManager.h"

namespace alphaone
{
OutstandingOrderManager::OutstandingOrderManager()
    : price_order_map_{std::map<BookPrice, std::pair<BookQty, OrderMap>, dynamic_compare>(
                           dynamic_compare(dynamic_compare::less)),
                       std::map<BookPrice, std::pair<BookQty, OrderMap>, dynamic_compare>(
                           dynamic_compare(dynamic_compare::greater))}
    , side_string_{"[Ask]", "[Bid]"}
    , share_potential_{0., 0.}
{
}

OutstandingOrderManager::~OutstandingOrderManager()
{
}

bool OutstandingOrderManager::InsertOrder(const Timestamp &event_loop_time, BookSide side,
                                          OrderNo orderno, BookPrice price, BookQty qty,
                                          const Timestamp &expired_time)
{
    auto [it, is_success] =
        orderno_map_.insert({orderno, {side, price, Timestamp::invalid()}});
    if (BRANCH_UNLIKELY(!is_success))
    {
        SPDLOG_WARN("[{}] duplicate orderno {}, going to skip add outstanding order", __func__,
                    orderno);
        return false;
    }
    auto pit = price_order_map_[side].find(price);
    if (BRANCH_UNLIKELY(pit == price_order_map_[side].end()))
    {
        pit = price_order_map_[side].insert({price, {0, OrderMap()}}).first;
    }
    pit->second.first += qty;
    pit->second.second.insert(
        {orderno,
         {price, qty, qty, expired_time, event_loop_time, Timestamp::invalid()}});
    share_potential_[side] += qty;
    return true;
}

bool OutstandingOrderManager::InsertCancelOrder(BookSide side, OrderNo orderno, BookPrice price)
{
    auto [it, is_sucess] =
        orderno_to_be_cancel_.insert({orderno, {side, price, Timestamp::invalid()}});
    if (BRANCH_UNLIKELY(!is_sucess))
    {
        SPDLOG_WARN("[{}] duplicate cancel orderno {}", __func__, orderno);
    }
    return is_sucess;
}

bool OutstandingOrderManager::RemoveOrder(OrderNo orderno)
{
    BookSide  side;
    BookPrice price;
    if (auto oit = orderno_map_.find(orderno); oit != orderno_map_.end())
    {
        side  = oit->second.side_;
        price = oit->second.price_;
        orderno_map_.erase(oit);
    }
    else
    {
        SPDLOG_WARN("[{}] Cannot find orderno {}, skip remove", __func__, orderno);
        return false;
    }

    auto pit = price_order_map_[side].find(price);
    if (pit == price_order_map_[side].end())
    {
        return false;
    }
    auto &order_map = pit->second.second;
    if (auto oit = order_map.find(orderno); (BRANCH_LIKELY(oit != order_map.end())))
    {
        pit->second.first -= oit->second.leaves_qty_;
        share_potential_[side] -= oit->second.leaves_qty_;
        order_map.erase(oit);
    }
    else
    {
        SPDLOG_WARN("[{}] Cannot find orderno = {} for price = {}", __func__, orderno, price);
        return false;
    }
    if (std::round(pit->second.first) <= 0.)
    {
        price_order_map_[side].erase(pit);
    }
    return true;
}

std::pair<bool, bool> OutstandingOrderManager::RemoveCancelOrder(OrderNo orderno,
                                                                 bool    is_clean_outstanding)
{
    auto it = orderno_to_be_cancel_.find(orderno);
    if (it == orderno_to_be_cancel_.end())
    {
        SPDLOG_DEBUG("[{}] Cannot find orderno = {} in orderno to be cancelled", __func__, orderno);
        return {false, false};
    }

    orderno_to_be_cancel_.erase(it);
    if (is_clean_outstanding)
    {
        return {true, RemoveOrder(orderno)};
    }
    return {true, false};
}

bool OutstandingOrderManager::UpdateOrder(OrderNo orderno, BookQty qty, bool is_change_qty)
{
    auto it = orderno_map_.find(orderno);
    if (it == orderno_map_.end())
    {
        SPDLOG_WARN("[{}] Cannot find orderno = {}", __func__, orderno);
        return false;
    }
    const auto side  = it->second.side_;
    const auto price = it->second.price_;
    auto       pit   = price_order_map_[side].find(price);
    if (pit == price_order_map_[side].end())
    {
        SPDLOG_WARN("[{}] Cannot find price = {}", __func__, price);
        return false;
    }
    auto &order_map = pit->second.second;
    if (auto oit = order_map.find(orderno); oit != order_map.end())
    {
        auto &order_left_qty = oit->second.leaves_qty_;
        order_left_qty += qty;
        share_potential_[side] += qty;
        if (BRANCH_UNLIKELY(is_change_qty))
        {
            oit->second.sent_qty_ += qty;
        }
        if (std::round(order_left_qty) <= 0.)
        {
            orderno_map_.erase(it);
            orderno_to_be_cancel_.erase(oit->first);
            order_map.erase(oit);
        }

        pit->second.first += qty;
        if (std::round(pit->second.first) <= 0.)
        {
            price_order_map_[side].erase(pit);
        }
        return true;
    }
    SPDLOG_WARN("[{}] Cannot find orderno {} at price = {}", __func__, orderno, price);
    return false;
}

void OutstandingOrderManager::CleanExpiredOrders(const Timestamp &event_loop_time)
{
    for (int side = Ask; side < AskBid; ++side)
    {
        for (auto &[price, qty_order_map_pair] : price_order_map_[side])
        {
            auto &order_map = qty_order_map_pair.second;
            for (auto oit = order_map.begin(); oit != order_map.end();)
            {
                if (event_loop_time > oit->second.expired_time_)
                {
                    SPDLOG_WARN("{} [{}] {} orderno = {} is expired!", event_loop_time, __func__,
                                side_string_[side], oit->first);
                    qty_order_map_pair.first -= oit->second.leaves_qty_;
                    share_potential_[side] -= oit->second.leaves_qty_;
                    orderno_map_.erase(oit->first);
                    oit = order_map.erase(oit);
                }
                else
                {
                    ++oit;
                }
            }
        }
    }
}

void OutstandingOrderManager::PrintOrders()
{
    for (int side = Ask; side < AskBid; ++side)
    {
        for (auto &[price, qty_order_map_pair] : price_order_map_[side])
        {
            for (auto &[orderno, outstanding_order] : qty_order_map_pair.second)
            {
                SPDLOG_INFO("[{}] {} {} {} {} {}", __func__, side_string_[side], price,
                            qty_order_map_pair.first, orderno, outstanding_order);
            }
        }
    }
}

BookQty OutstandingOrderManager::GetQtyFromPrice(BookSide side, BookPrice price)
{
    if (auto pit = price_order_map_[side].find(price); pit != price_order_map_[side].end())
    {
        return pit->second.first;
    }
    return 0.;
}

BookQty OutstandingOrderManager::GetSharesPotential(BookSide side)
{
    return share_potential_[side];
}

OrderMap *OutstandingOrderManager::GetOrderMapFromPrice(BookSide side, BookPrice price)
{
    if (auto pit = price_order_map_[side].find(price); pit != price_order_map_[side].end())
    {
        return &pit->second.second;
    }
    return nullptr;
}

PriceOrderMap *OutstandingOrderManager::GetPriceOrderMapFromSide(BookSide side)
{
    return &price_order_map_[side];
}

FlatOrderMap *OutstandingOrderManager::GetCancellingMap()
{
    return &orderno_to_be_cancel_;
}

FlatOrderMap *OutstandingOrderManager::GetOrderMap()
{
    return &orderno_map_;
}

}  // namespace alphaone
