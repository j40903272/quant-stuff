#include "TWSEFileBook.h"

#include "infrastructure/common/math/Math.h"
#include "infrastructure/common/spdlog/fmt/bundled/color.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/util/Logger.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_TWSEDataFile.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>

namespace alphaone
{
TWSEFileBook::TWSEFileBook(const Symbol *symbol, OrderFactory *order_factory,
                           LevelFactory *level_factory)
    : MarketByOrderBook{symbol, order_factory, level_factory, false, false,
                        false,  false,         false,         false}
    , twse_hack_multiplier_{63UL * 63UL * 63UL * 63UL * 63UL}
    , twse_hack_space_{62UL * twse_hack_multiplier_}
    , twse_hack_closing_time_{Timestamp::invalid()}
    , twse_hack_opening_time_{Timestamp::invalid()}
    , last_provider_time_{Timestamp::invalid()}
    , trade_pride_{0.}
{
}

void TWSEFileBook::OnAdd(const MarketDataMessage *mm)
{
    if (twse_hack_opening_time_.is_valid() && mm->provider_time > twse_hack_opening_time_)
    {
        UncrossMarketByOrderBook(mm);
        trade_pride_ = 0.;
    }

    message_add_->ConstructBookMessageFromMarketMessage(mm);

    const auto &mboprice{mm->GetMboPrice()};
    const auto &order_id{mm->mbo.order_id};
    auto *      order{order_factory_->CreateOrder(order_id, mm->provider_time, mm->mbo.qty)};
    map_id_to_orders_[order_id].push_back(order);

    if (mm->mbo.side == BID)
    {
        auto [it, success] = all_bids_.insert({mboprice, nullptr});
        if (success)  // add new level
        {
            auto *level{level_factory_->CreateLevel(order, mboprice, mm->mbo.nord, BID)};
            it->second = level;
            if (mboprice == all_bids_.rbegin()->first)
            {
                CreateLevel(level, order);
            }
            else
            {
                CreateLevel(level, order, (++it)->second);
            }
        }
        else  // found existing level
        {
            CreateOrder(it->second, order);
        }
    }
    else
    {
        auto [it, success] = all_asks_.insert({mboprice, nullptr});
        if (success)  // add new level
        {
            auto *level{level_factory_->CreateLevel(order, mboprice, mm->mbo.nord, ASK)};
            it->second = level;
            if (mboprice == all_asks_.rbegin()->first)
            {
                CreateLevel(level, order);
            }
            else
            {
                CreateLevel(level, order, (++it)->second);
            }
        }
        else  // found existing level
        {
            CreateOrder(it->second, order);
        }
    }

    last_provider_time_ = mm->provider_time;
    ++book_statistic_adds_total_;
}

void TWSEFileBook::OnDelete(const MarketDataMessage *mm)
{
    if (twse_hack_opening_time_.is_valid() && mm->provider_time > twse_hack_opening_time_)
    {
        UncrossMarketByOrderBook(mm);
        trade_pride_ = 0.;
    }

    message_delete_->ConstructBookMessageFromMarketMessage(mm);

    const auto &mboprice{mm->GetMboPrice()};
    const auto &order_id{mm->mbo.order_id};

    auto io{map_id_to_orders_.find(order_id)};
    if (io == map_id_to_orders_.end())
    {
        SPDLOG_WARN("[{}] {} unrecognized order of id={}", __func__, mm->provider_time,
                    MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
    }
    else
    {
        auto &orders{io->second};
        auto  it{std::find_if(orders.begin(), orders.end(),
                             [mboprice](const Order *o)
                             { return o->level_ != nullptr && o->GetPrice() == mboprice; })};

        if (it == orders.end())
        {
            it = std::find_if(orders.begin(), orders.end(),
                              [](const Order *o) { return o->level_ == nullptr; });
        }

        if (it == orders.end())
        {
            it = orders.begin();
        }

        if (it != orders.end())
        {
            auto &order{*it};
            auto &level{order->level_};
            if (level != nullptr)
            {
                RemoveOrder(level, order);
            }
            else
            {
                SPDLOG_WARN("{} trying to delete order of empty level with id={}",
                            mm->provider_time,
                            MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
            }
            orders.erase(it);
        }
    }

    last_provider_time_ = mm->provider_time;
    ++book_statistic_deletes_total_;
}

void TWSEFileBook::OnModifyWithPrice(const MarketDataMessage *mm)
{
    if (twse_hack_opening_time_.is_valid() && mm->provider_time > twse_hack_opening_time_)
    {
        UncrossMarketByOrderBook(mm);
        trade_pride_ = 0.;
    }

    const auto &mboprice{mm->GetMboPrice()};
    const auto &mboqty{mm->mbo.qty};
    const auto &order_id{mm->mbo.order_id};

    auto io{map_id_to_orders_.find(order_id)};
    if (io == map_id_to_orders_.end())
    {
        SPDLOG_WARN("[{}] {} unrecognized order of id={}", __func__, mm->provider_time,
                    MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
    }
    else
    {
        auto &orders{io->second};
        auto  it{std::find_if(orders.begin(), orders.end(),
                             [mboqty](const Order *o) { return o->qty_ == mboqty; })};

        if (it == orders.end())
        {
            it = orders.begin();
        }

        if (it != orders.end())
        {
            auto &order{*it};
            auto &level{order->level_};
            if (level != nullptr)
            {
                message_delete_->ConstructBookMessageFromMarketMessage(mm);
                RemoveOrder(level, order);
            }
            else
            {
                SPDLOG_WARN("[{}] {} modify order price upon empty level for order of id={}",
                            __func__, mm->provider_time,
                            MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
            }
            orders.erase(it);
        }
    }

    message_add_->ConstructBookMessageFromMarketMessage(mm);
    OnAdd(mm);

    last_provider_time_ = mm->provider_time;
    ++book_statistic_modify_with_price_total_;
}

void TWSEFileBook::OnModifyWithQty(const MarketDataMessage *mm)
{
    if (twse_hack_opening_time_.is_valid() && mm->provider_time > twse_hack_opening_time_)
    {
        UncrossMarketByOrderBook(mm);
        trade_pride_ = 0.;
    }

    const auto &mboprice{mm->GetMboPrice()};
    const auto &mboqty{mm->mbo.qty};
    const auto &order_id{mm->mbo.order_id};

    auto io{map_id_to_orders_.find(order_id)};
    if (io == map_id_to_orders_.end())
    {
        SPDLOG_WARN("[{}] {} unrecognized order of id={}", __func__, mm->provider_time,
                    MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
    }
    else
    {
        auto &orders{io->second};
        auto  it{std::find_if(orders.begin(), orders.end(),
                             [mboprice](const Order *o)
                             { return o->level_ != nullptr && o->GetPrice() == mboprice; })};

        if (it == orders.end())
        {
            it = std::find_if(orders.begin(), orders.end(),
                              [](const Order *o) { return o->level_ == nullptr; });
        }

        if (it == orders.end())
        {
            it = orders.begin();
        }

        if (it != orders.end())
        {
            auto &order{*it};
            auto &level{order->level_};
            if (level != nullptr)
            {
                if (mboqty < 0.0)
                {
                    message_delete_->ConstructBookMessageFromMarketMessage(mm);
                    DecreaseOrderQty(level, order, std::abs(mboqty));
                    if (order->qty_ <= 0.0)
                    {
                        RemoveOrder(level, order);
                        orders.erase(it);
                    }
                }
                else if (mboqty > 0.0)
                {
                    message_add_->ConstructBookMessageFromMarketMessage(mm);
                    IncreaseOrderQty(level, order, mboqty);
                }
            }
            else
            {
                if (mboqty < 0.0)
                {
                    SPDLOG_WARN("[{}] {} decrease order qty upon empty level for order of id={}",
                                __func__, mm->provider_time,
                                MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
                }
                else if (mboqty > 0.0)
                {
                    SPDLOG_WARN("[{}] {} increase order qty upon empty level for order of id={}",
                                __func__, mm->provider_time,
                                MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
                }
            }
        }
    }

    last_provider_time_ = mm->provider_time;
    ++book_statistic_modify_with_qty_total_;
}

void TWSEFileBook::OnTrade(const MarketDataMessage *mm)
{
    if (twse_hack_opening_time_.is_valid() && mm->provider_time > twse_hack_opening_time_ &&
        mm->provider_time > last_provider_time_)
    {
        UncrossMarketByOrderBook(mm);
    }

    message_trade_->ConstructBookMessageFromMarketMessage(mm);
    message_trade_->trade.counterparty_order_id = mm->trade.counterparty_order_id;
    twse_hack_opening_time_                     = mm->provider_time;

    const auto &tradeprice{mm->GetTradePrice()};
    const auto &tradeqty{mm->GetTradeQty()};
    auto        im{map_id_to_orders_.find(mm->trade.order_id)};
    const auto &order_id{im != map_id_to_orders_.end() && !im->second.empty()
                             ? mm->trade.order_id
                             : GetTWSEOtherOrderId(mm->trade.order_id)};

    auto io{map_id_to_orders_.find(order_id)};
    if (io == map_id_to_orders_.end())
    {
        SPDLOG_WARN("[{}] {} unrecognized order of id={}", __func__, mm->provider_time,
                    MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
    }
    else
    {
        auto &orders{io->second};
        auto  it{std::find_if(orders.begin(), orders.end(),
                             [tradeprice](const Order *o)
                             { return o->level_ != nullptr && o->GetPrice() == tradeprice; })};

        if (it == orders.end())  // cannot find same price, find market price order (price=0.0)
        {
            it = std::find_if(orders.begin(), orders.end(),
                              [](const Order *o)
                              { return o->level_ != nullptr && o->GetPrice() == 0.0; });
        }

        if (it == orders.end())
        {
            it = std::find_if(orders.begin(), orders.end(),
                              [](const Order *o) { return o->level_ == nullptr; });
        }

        if (it == orders.end())
        {
            it = orders.begin();
        }

        if (it != orders.end())
        {
            auto &order{*it};
            auto &level{order->level_};
            order->qty_ -= tradeqty;
            if (level != nullptr)
            {
                level->qty_ -= tradeqty;
                if (order->qty_ <= 0.0)
                {
                    RemoveOrder(level, order, false);
                }
            }
            else
            {
                order->qty_ = 0.0;
            }
            if (order->qty_ <= 0.0)
            {
                orders.erase(it);
            }

            last_trade_time_  = mm->provider_time;
            last_trade_price_ = tradeprice;
            last_trade_qty_   = tradeqty;
            trade_pride_      = tradeprice;

            EmitPostTrade();
        }
    }

    ++book_statistic_trades_total_;
}

void TWSEFileBook::OnSnapshot(const MarketDataMessage *mm)
{
    ++book_statistic_snapshots_total_;
}

void TWSEFileBook::OnPacketEnd(const MarketDataMessage *mm)
{
    const Timestamp exchange_time{message_packet_end_->exchange_time};
    message_packet_end_->ConstructBookMessageFromMarketMessage(mm);
    message_packet_end_->exchange_time = exchange_time;

    EmitPacketEnd();

    ++book_statistic_packet_ends_total_;

    if (BRANCH_LIKELY(touch_ask_ != nullptr && touch_bid_ != nullptr))
    {
        prev_touch_price_[Ask] = touch_ask_->price_;
        prev_touch_price_[Bid] = touch_bid_->price_;
    }
}

void TWSEFileBook::OnSparseStop()
{
    EmitSparseStop();
}

void TWSEFileBook::RemoveOrder(Level *level, Order *order, const bool emit)
{
    if (order->next_)
    {
        order->next_->prev_ = order->prev_;
    }

    if (order->prev_)
    {
        order->prev_->next_ = order->next_;
    }

    if (order == level->tail_)
    {
        level->tail_ = order->next_;
    }

    if (order == level->head_)
    {
        level->head_ = order->prev_;
    }

    level->qty_ -= order->qty_;
    level->nord_ -= 1;

    if (BRANCH_UNLIKELY(message_delete_->GetMarketByOrderPrice() != level->price_))
    {
        message_delete_->SetMarketByOrderPrice(level->price_);
    }

    if (BRANCH_UNLIKELY(message_delete_->GetMarketByOrderSide() != level->side_))
    {
        message_delete_->SetMarketByOrderSide(level->side_);
    }

    message_delete_->mbo.qty                 = order->qty_;
    message_delete_->mbo.nord                = 1;
    message_delete_->order_                  = order;
    message_delete_->is_level_fully_deleted_ = true;

    if (emit)
    {
        EmitPreDelete();
    }

    order_factory_->DeleteOrder(order);
    if (level->nord_ == 0 || level->qty_ == 0.)
    {
        RemoveLevel(level);
    }

    if (emit)
    {
        EmitPostDelete();
    }
}

void TWSEFileBook::ClearOrders(Level *level)
{
    Order *     order{level->head_};
    Order *     tmp{nullptr};
    const auto &price{level->price_};

    while (order != nullptr)
    {
        const auto &order_id{order->id_};

        tmp   = order;
        order = order->prev_;

        auto io{map_id_to_orders_.find(order_id)};
        if (io == map_id_to_orders_.end())
        {
            SPDLOG_DEBUG("[{}] {} heuristically remove unrecognized order with id={}", __func__,
                         last_event_loop_time_,
                         MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));
        }
        else
        {
            SPDLOG_DEBUG("[{}] {} heuristically remove recognized order with id={}", __func__,
                         last_event_loop_time_,
                         MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(order_id));

            auto &orders{io->second};
            auto  it{std::find_if(orders.begin(), orders.end(),
                                 [price](const Order *o) { return o->GetPrice() == price; })};

            if (it != orders.end())
            {
                orders.erase(it);
            }
        }

        RemoveOrder(level, tmp);
    }
}

ExternalOrderId TWSEFileBook::GetTWSEOtherOrderId(const ExternalOrderId order_id)
{
    // this function change order number of twse order id from 'ZA000' to 'Z 000'
    // TSE odr file use space(' ') replace some order id's second char(Z' '000),
    // but mth file use original(Z'A'000).
    // twse order id = brocker code (4 char) + order id (5 char)
    const ExternalOrderId second =
        ((order_id / twse_hack_multiplier_) % 63UL) * twse_hack_multiplier_;
    return order_id - second + twse_hack_space_;
}

void TWSEFileBook::UncrossMarketByOrderBook(const MarketDataMessage *mm)
{
    {
        const BookPrice equilibrium_price{
            mm->market_data_message_type == MarketDataMessageType_Trade ? mm->GetTradePrice()
                                                                        : trade_pride_};

        if (equilibrium_price == 0.)
        {
            return;
        }

        message_delete_->ConstructBookMessageFromMarketMessage(mm);

        Level *tmp{nullptr};

        Level *bid{touch_bid_};
        while (bid != nullptr && equilibrium_price < bid->price_)
        {
            tmp                        = bid->next_;
            message_delete_->mbo.side  = bid->side_;
            message_delete_->mbo.price = bid->price_;
            SPDLOG_DEBUG("{} trying to uncross bid with price={} qty={}", mm->provider_time,
                         bid->price_, bid->qty_);
            ClearOrders(bid);
            bid = tmp;
            ++book_statistic_uncross_book_total_;
        }

        Level *ask{touch_ask_};
        while (ask != nullptr && equilibrium_price > ask->price_)
        {
            tmp                        = ask->next_;
            message_delete_->mbo.side  = ask->side_;
            message_delete_->mbo.price = ask->price_;
            SPDLOG_DEBUG("{} trying to uncross ask with price={} qty={}", mm->provider_time,
                         ask->price_, ask->qty_);
            ClearOrders(ask);
            ask = tmp;
            ++book_statistic_uncross_book_total_;
        }
    }
}

void TWSEFileBook::EmitPreDelete() const
{
    for (auto &listener : listeners_pre_)
    {
        listener->OnPreBookDelete(last_event_loop_time_, message_delete_);
    }
}

void TWSEFileBook::EmitPostAdd() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookAdd(last_event_loop_time_, message_add_);
    }
}

void TWSEFileBook::EmitPostDelete() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookDelete(last_event_loop_time_, message_delete_);
    }
}

void TWSEFileBook::EmitPostModifyWithPrice() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookModifyWithPrice(last_event_loop_time_, message_modify_with_price_);
    }
}

void TWSEFileBook::EmitPostModifyWithQty() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookModifyWithQty(last_event_loop_time_, message_modify_with_qty_);
    }
}

void TWSEFileBook::EmitPostSnapshot() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookSnapshot(last_event_loop_time_, message_snapshot_);
    }
}

void TWSEFileBook::EmitPostTrade() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnTrade(last_event_loop_time_, message_trade_);
    }
}

void TWSEFileBook::EmitPacketEnd() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnPacketEnd(last_event_loop_time_, message_packet_end_);
    }
}

void TWSEFileBook::EmitSparseStop() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnSparseStop(message_sparse_stop_->event_loop_time_, message_sparse_stop_);
    }
}
}  // namespace alphaone
