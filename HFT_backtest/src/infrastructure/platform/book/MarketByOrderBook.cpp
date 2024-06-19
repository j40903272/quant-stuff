#include "MarketByOrderBook.h"

#include "infrastructure/common/math/Math.h"
#include "infrastructure/common/spdlog/fmt/bundled/color.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/util/Logger.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>

namespace alphaone
{
MarketByOrderBook::MarketByOrderBook(const Symbol *symbol, OrderFactory *order_factory,
                                     LevelFactory *level_factory,
                                     const bool    is_trading_through_delete,
                                     const bool is_touching_on_delete, const bool is_queue_like,
                                     const bool is_overrunning_checking,
                                     const bool is_rejoining_checking,
                                     const bool is_emit_allowed_only_when_valid)
    : Book{symbol, is_emit_allowed_only_when_valid}
    , touch_bid_{nullptr}
    , touch_ask_{nullptr}
    , last_boundary_bid_{INVALID_BID_PRICE}
    , last_boundary_ask_{INVALID_ASK_PRICE}
    , is_queue_like_{is_queue_like}
    , order_factory_{order_factory}
    , level_factory_{level_factory}
    , is_trade_{false}
    , last_trade_price_{0.0}
    , last_trade_qty_{0.0}
    , last_trade_side_{BID}
    , book_statistic_uncross_book_total_{0}
    , book_statistic_unlock_book_total_{0}
    , is_trading_through_delete_{is_trading_through_delete}
    , is_touching_on_delete_{is_touching_on_delete}
    , is_overrunning_checking_{is_overrunning_checking}
    , is_rejoining_checking_{is_rejoining_checking}
    , market_data_message_delete_{DataSourceType::MarketByOrder}
    , market_order_qty_{0.0, 0.0}
{
    market_data_message_delete_.symbol = symbol;
    market_data_message_delete_.market_data_message_type ==
        MarketDataMessageType::MarketDataMessageType_Delete;
}

MarketByOrderBook::MarketByOrderBook(const nlohmann::json &json, const Symbol *symbol,
                                     OrderFactory *order_factory, LevelFactory *level_factory)
    : MarketByOrderBook::MarketByOrderBook(
          symbol, order_factory, level_factory, json.value("is_trading_through_delete", false),
          json.value("is_touching_on_delete", false), json.value("is_queue_like", false),
          json.value("is_overrunning_checking", false), json.value("is_rejoining_checking", false),
          json.value("is_emit_allowed_only_when_valid", true))
{
}

MarketByOrderBook::MarketByOrderBook(const GlobalConfiguration *configuration, const Symbol *symbol,
                                     OrderFactory *order_factory, LevelFactory *level_factory)
    : MarketByOrderBook::MarketByOrderBook(
          configuration->GetJson().value("MarketByOrderBook", nlohmann::json({})), symbol,
          order_factory, level_factory)
{
}

MarketByOrderBook::~MarketByOrderBook()
{
    const auto &symbol_str = GetSymbol() != nullptr ? "[" + GetSymbol()->to_string() + "]" : "";
    SPDLOG_INFO("{} [{}] {} #ADDs={} #DELs={} #MWPs={} #MWQs={} #SNSs={} #TRDs={} #PKEs={}",
                last_event_loop_time_, __func__, symbol_str, GetNumberAdds(), GetNumberDeletes(),
                GetNumberModifyWithPrices(), GetNumberModifyWithQtys(), GetNumberSnapshots(),
                GetNumberTrades(), GetNumberPacketEnds());

    Done();
}

void MarketByOrderBook::Done()
{
    listeners_pre_.clear();
    listeners_post_.clear();
    Reset();
}

BookPrice MarketByOrderBook::GetPrice(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        if (BRANCH_LIKELY(nolv < all_bids_.size()))
        {
            return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->price_;
        }
        return INVALID_BID_PRICE;
    }
    else
    {
        if (BRANCH_LIKELY(nolv < all_asks_.size()))
        {
            return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->price_;
        }
        return INVALID_ASK_PRICE;
    }
}

BookPrice MarketByOrderBook::GetPrice(BidType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_bids_.size()))
    {
        return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->price_;
    }
    return INVALID_BID_PRICE;
}

BookPrice MarketByOrderBook::GetPrice(AskType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_asks_.size()))
    {
        return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->price_;
    }
    return INVALID_ASK_PRICE;
}

BookQty MarketByOrderBook::GetQty(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        if (BRANCH_LIKELY(nolv < all_bids_.size()))
        {
            return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->qty_;
        }
    }
    else
    {
        if (BRANCH_LIKELY(nolv < all_asks_.size()))
        {
            return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->qty_;
        }
    }

    return -1.0;
}

BookPrice MarketByOrderBook::GetQty(BidType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_bids_.size()))
    {
        return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->qty_;
    }
    return -1.0;
}

BookPrice MarketByOrderBook::GetQty(AskType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_asks_.size()))
    {
        return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->qty_;
    }
    return -1.0;
}

BookNord MarketByOrderBook::GetNord(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        if (BRANCH_LIKELY(nolv < all_bids_.size()))
        {
            return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->nord_;
        }
    }
    else
    {
        if (BRANCH_LIKELY(nolv < all_asks_.size()))
        {
            return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->nord_;
        }
    }

    return 0;
}

BookNord MarketByOrderBook::GetNord(BidType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_bids_.size()))
    {
        return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->nord_;
    }
    return 0;
}

BookNord MarketByOrderBook::GetNord(AskType *, const BookNolv nolv) const
{
    if (BRANCH_LIKELY(nolv < all_asks_.size()))
    {
        return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->nord_;
    }
    return 0;
}

BookNolv MarketByOrderBook::GetNolv(const BookSide side) const
{
    return side == BID ? all_bids_.size() : all_asks_.size();
}

BookPrice MarketByOrderBook::GetMidPrice() const
{
    return 0.5 * (GetPrice(BidType::GetType()) + GetPrice(AskType::GetType()));
}

BookPrice MarketByOrderBook::GetWeightedPrice() const
{
    return (GetQty(AskType::GetType()) * GetPrice(BidType::GetType()) +
            GetQty(BidType::GetType()) * GetPrice(AskType::GetType())) /
           (GetQty(BidType::GetType()) + GetQty(AskType::GetType()));
}

BookQty MarketByOrderBook::GetMidQty() const
{
    return 0.5 * (GetQty(BidType::GetType()) + GetQty(AskType::GetType()));
}

BookQty MarketByOrderBook::GetMarketOrderQty(const BookSide side) const
{
    return market_order_qty_[side];
}

BookPrice MarketByOrderBook::GetSpread(const BookNolv bid_level, const BookNolv ask_level) const
{
    if (BRANCH_UNLIKELY(all_bids_.size() <= bid_level or all_asks_.size() <= ask_level))
    {
        return static_cast<BookPrice>(NaN);
    }

    return (GetPrice(AskType::GetType(), ask_level) - GetPrice(BidType::GetType(), bid_level));
}

BookPrice MarketByOrderBook::GetSpreadAsPercentOfMid(const BookNolv b_lvl,
                                                     const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(all_bids_.size() <= b_lvl or all_asks_.size() <= a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }

    return (GetPrice(AskType::GetType(), a_lvl) - GetPrice(BidType::GetType(), b_lvl)) /
           GetMidPrice();
}

BookPrice MarketByOrderBook::GetSpreadAsPercentOfBid(const BookNolv b_lvl,
                                                     const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(all_bids_.size() <= b_lvl or all_asks_.size() <= a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }

    return (GetPrice(AskType::GetType(), a_lvl) - GetPrice(BidType::GetType(), b_lvl)) /
           GetPrice(BidType::GetType(), b_lvl);
}

BookPrice MarketByOrderBook::GetSpreadAsReturn(const BookNolv b_lvl, const BookNolv a_lvl) const
{
    if (all_bids_.size() <= b_lvl or all_asks_.size() <= a_lvl)
    {
        return static_cast<BookPrice>(NaN);
    }

    return y_log(GetPrice(AskType::GetType(), a_lvl) / GetPrice(BidType::GetType(), b_lvl));
}

BookPrice MarketByOrderBook::GetPriceBehindQty(const BookSide &side, const BookQty &qty) const
{
    BookQty total_qty{0};
    if (side == BID)
    {
        auto it{all_bids_.rbegin()};
        while (it != all_bids_.rend())
        {
            total_qty += it->second->qty_;
            if (total_qty >= qty)
            {
                return it->second->price_;
            }
            ++it;
        }
        return all_bids_.begin()->second->price_;
    }
    else
    {
        auto it(all_asks_.rbegin());
        while (it != all_asks_.rend())
        {
            total_qty += it->second->qty_;
            if (total_qty >= qty)
            {
                return it->second->price_;
            }
            ++it;
        }
        return all_asks_.begin()->second->price_;
    }
}

BookPrice MarketByOrderBook::GetPriceBeforeQty(const BookSide &side, const BookQty &qty) const
{
    BookQty total_qty{0.0};
    if (side == BID)
    {
        auto it{all_bids_.rbegin()};
        while (it != all_bids_.rend())
        {
            total_qty += it->second->qty_;
            if (total_qty > qty)
            {
                return (it == all_bids_.rbegin())
                           ? it->second->price_ + symbol_->GetTickSize(it->second->price_, true)
                           : (--it)->second->price_;
            }
            ++it;
        }
        return all_bids_.begin()->second->price_;
    }
    else
    {
        auto it{all_asks_.rbegin()};
        while (it != all_asks_.rend())
        {
            total_qty += it->second->qty_;
            if (total_qty > qty)
            {
                return (it == all_asks_.rbegin())
                           ? it->second->price_ - symbol_->GetTickSize(it->second->price_, false)
                           : (--it)->second->price_;
            }
            ++it;
        }
        return all_asks_.begin()->second->price_;
    }
}

BookQty MarketByOrderBook::GetQtyBehindPrice(const BookSide &side, const BookPrice &price) const
{
    BookQty total_qty{0.0};
    if (side == BID)
    {
        for (auto it{all_bids_.rbegin()}; it != all_bids_.rend(); ++it)
        {
            if (it->second->price_ >= price)
            {
                total_qty += it->second->qty_;
            }
            else
            {
                return total_qty;
            }
        }
        return total_qty;
    }
    else
    {
        for (auto it{all_asks_.rbegin()}; it != all_asks_.rend(); ++it)
        {
            if (it->second->price_ <= price)
            {
                total_qty += it->second->qty_;
            }
            else
            {
                return total_qty;
            }
        }
        return total_qty;
    }
}

BookPrice MarketByOrderBook::GetPriceBehindLevel(const BookPrice &price, const BookSide &side,
                                                 const BookNolv &nolv) const
{
    if (BRANCH_LIKELY(nolv > 0))
    {
        if (side == BID)
        {
            BookPrice result_price{price};
            for (unsigned int i{1}; i < nolv; ++i)
            {
                result_price -= GetSymbol()->GetTickSize(result_price, false);
            }
            return std::max(0.0, result_price);
        }
        else
        {
            BookPrice result_price{price};
            for (unsigned int i{1}; i < nolv; ++i)
            {
                result_price += GetSymbol()->GetTickSize(result_price, true);
            }
            return result_price;
        }
    }
    else
    {
        return price;
    }
}

void MarketByOrderBook::OnAdd(const MarketDataMessage *mm)
{
    message_add_->ConstructBookMessageFromMarketMessage(mm);

    const auto &mbomsg{mm->mbo};
    const auto &mboprice{mm->GetMboPrice()};

    if (mbomsg.side == BID)
    {
        auto order{order_factory_->CreateOrder(mbomsg.order_id, mm->provider_time, mbomsg.qty)};
        auto [it, success]{all_bids_.insert({mboprice, nullptr})};
        if (success)  // add new level
        {
            it->second = level_factory_->CreateLevel(order, mboprice, mbomsg.nord, BID);
            if (mboprice == all_bids_.rbegin()->first)
            {
                CreateLevel(it->second, order);
            }
            else
            {
                CreateLevel(it->second, order, (++it)->second);
            }
        }
        else  // found existing level
        {
            CreateOrder(it->second, order);
        }
    }
    else
    {
        auto order{order_factory_->CreateOrder(mbomsg.order_id, mm->provider_time, mbomsg.qty)};
        auto [it, success]{all_asks_.insert({mboprice, nullptr})};
        if (success)  // add new level
        {
            it->second = level_factory_->CreateLevel(order, mboprice, mbomsg.nord, ASK);
            if (mboprice == all_asks_.rbegin()->first)
            {
                CreateLevel(it->second, order);
            }
            else
            {
                CreateLevel(it->second, order, (++it)->second);
            }
        }
        else  // found existing level
        {
            CreateOrder(it->second, order);
        }
    }

    ++book_statistic_adds_total_;

    UncrossMarketByOrderBook(mm);
}

void MarketByOrderBook::OnDelete(const MarketDataMessage *mm)
{
    message_delete_->ConstructBookMessageFromMarketMessage(mm);

    const auto &mboprice{mm->GetMboPrice()};
    const auto &is_trade{mm->market_data_message_type == MarketDataMessageType_Trade};

    if (mm->mbo.side == BID)
    {
        auto it{all_bids_.find(mboprice)};
        if (BRANCH_LIKELY(it != all_bids_.end()))
        {
            RemoveQty(it->second, mm->mbo.qty, is_trade);
        }
    }
    else
    {
        auto it{all_asks_.find(mboprice)};
        if (BRANCH_LIKELY(it != all_asks_.end()))
        {
            RemoveQty(it->second, mm->mbo.qty, is_trade);
        }
    }

    ++book_statistic_deletes_total_;
}

void MarketByOrderBook::OnModifyWithPrice(const MarketDataMessage *mm)
{
    ++book_statistic_modify_with_price_total_;
}

void MarketByOrderBook::OnModifyWithQty(const MarketDataMessage *mm)
{
    ++book_statistic_modify_with_qty_total_;
}

BookSide MarketByOrderBook::ParseTradeSide(const BookPrice bid_price, const BookPrice ask_price,
                                           const BookPrice trade_price)
{
    if (trade_price >= ask_price)
    {
        return BID;
    }
    else if (trade_price <= bid_price)
    {
        return ASK;
    }

    const BookPrice mid_price{0.5 * (bid_price + ask_price)};
    if (trade_price > mid_price)
    {
        return BID;
    }
    else if (trade_price < mid_price)
    {
        return ASK;
    }
    else
    {
#ifdef AlphaOneDebug
        std::cout << "BID=" << bid_price << " ASK=" << ask_price << " TRD=" << trade_price << '\n';
#endif
        return ASK;
    }
}

void MarketByOrderBook::OnTrade(const MarketDataMessage *mm)
{
    message_trade_->ConstructBookMessageFromMarketMessage(mm);

    SPDLOG_INFO("am i here?10");
    if (mm->symbol->GetDataSourceID() < DataSourceID::END)
    {
        if (touch_bid_ != nullptr && touch_ask_ != nullptr)
        {
            const BookPrice bid_price{touch_bid_->price_};
            const BookPrice ask_price{touch_ask_->price_};
            message_trade_->trade.side = ParseTradeSide(bid_price, ask_price, mm->GetTradePrice());
        }
    }

    message_trade_->trade.counterparty_order_id = mm->trade.counterparty_order_id;

    is_trade_         = true;
    last_trade_time_  = mm->provider_time;
    last_trade_price_ = mm->GetTradePrice();
    last_trade_qty_   = mm->GetTradeQty();
    last_trade_side_  = message_trade_->trade.side;

    EmitPostTrade();

    // Trading-through levels are cleared
    if (is_trading_through_delete_)
    {
        CleanMarketByOrderBookAfterTrade(message_trade_->trade.side, mm);

        // Touching-on levels are removed
        if (is_touching_on_delete_)
        {
            market_data_message_delete_.market_data_message_type = MarketDataMessageType_Trade;
            market_data_message_delete_.provider_time            = mm->provider_time;
            market_data_message_delete_.exchange_time            = mm->exchange_time;
            market_data_message_delete_.sequence_number          = mm->sequence_number;
            market_data_message_delete_.mbo.price                = mm->trade.price;
            market_data_message_delete_.mbo.qty                  = mm->GetTradeQty();
            market_data_message_delete_.mbo.side                 = !message_trade_->trade.side;
            OnDelete(&market_data_message_delete_);
        }

        market_data_message_delete_.market_data_message_type = MarketDataMessageType_Delete;
    }

    ++book_statistic_trades_total_;

    return;
}

void MarketByOrderBook::ParseSnapshotOnBid(const MarketDataMessage *mm,
                                           const ExternalOrderId &order_id, BookPrice price,
                                           BookQty qty, BookNord nord)
{
    auto [it, success] = all_bids_.insert({price, nullptr});
    if (success)
    {
        if (qty != 0.)
        {
            Order *order{order_factory_->CreateOrder(order_id, mm->provider_time, qty)};
            Level *level{level_factory_->CreateLevel(order, price, nord, BID)};
            message_add_->ConstructBookMessageFromMarketMessage(mm);
            message_add_->mbo.price = price;
            message_add_->mbo.side  = BID;
            it->second              = level;
            if (price == all_bids_.rbegin()->first)
                CreateLevel(level, order);
            else
                CreateLevel(level, order, (++it)->second);
        }
        else
        {
            all_bids_.erase(it);
        }
        return;
    }

    const auto &qty_of_existing_level{it->second->qty_};
    if (qty_of_existing_level < qty)  // if existing level is smaller, then we need to add
    {
        message_add_->ConstructBookMessageFromMarketMessage(mm);
        message_add_->mbo.price = price;
        message_add_->mbo.side  = BID;
        CreateOrder(it->second, order_factory_->CreateOrder(order_id, mm->provider_time,
                                                            qty - qty_of_existing_level));
        return;
    }

    if (qty_of_existing_level > qty)  // if existing level is bigger, then we need to delete
    {
        message_delete_->ConstructBookMessageFromMarketMessage(mm);
        message_delete_->mbo.price = price;
        message_delete_->mbo.side  = BID;
        RemoveQty(it->second, qty_of_existing_level - qty);
        return;
    }
}

void MarketByOrderBook::ParseSnapshotOnAsk(const MarketDataMessage *mm,
                                           const ExternalOrderId &order_id, BookPrice price,
                                           BookQty qty, BookNord nord)
{
    auto [it, success] = all_asks_.insert({price, nullptr});
    if (success)
    {
        if (qty != 0.)
        {
            Order *order{order_factory_->CreateOrder(order_id, mm->provider_time, qty)};
            Level *level{level_factory_->CreateLevel(order, price, nord, ASK)};
            message_add_->ConstructBookMessageFromMarketMessage(mm);
            message_add_->mbo.price = price;
            message_add_->mbo.side  = ASK;
            it->second              = level;
            if (price == all_asks_.rbegin()->first)
                CreateLevel(level, order);
            else
                CreateLevel(level, order, (++it)->second);
        }
        else
        {
            all_asks_.erase(it);
        }
        return;
    }

    const auto &qty_of_existing_level{it->second->qty_};
    if (qty_of_existing_level < qty)  // if existing level is smaller, then we need to add
    {
        message_add_->ConstructBookMessageFromMarketMessage(mm);
        message_add_->mbo.price = price;
        message_add_->mbo.side  = ASK;
        CreateOrder(it->second, order_factory_->CreateOrder(order_id, mm->provider_time,
                                                            qty - qty_of_existing_level));
        return;
    }

    if (qty_of_existing_level > qty)  // if existing level is bigger, then we need to delete
    {
        message_delete_->ConstructBookMessageFromMarketMessage(mm);
        message_delete_->mbo.price = price;
        message_delete_->mbo.side  = ASK;
        RemoveQty(it->second, qty_of_existing_level - qty);
        return;
    }
}

void MarketByOrderBook::ParseMarketByPriceEntry(const MarketDataMessage *mm)
{
    auto &touch_index{market_by_price_entry_.touch_index_};
    auto &worst_index{market_by_price_entry_.worst_index_};
    auto &touch_price{market_by_price_entry_.touch_price_};
    auto &worst_price{market_by_price_entry_.worst_price_};

    touch_index[Bid] = 0;
    worst_index[Bid] = mm->mbp.count;
    touch_price[Bid] = mm->GetMbpBidPrice(0);
    worst_price[Bid] = mm->GetMbpBidPrice(mm->mbp.count - 1);
    touch_index[Ask] = 0;
    worst_index[Ask] = mm->mbp.count;
    touch_price[Ask] = mm->GetMbpAskPrice(0);
    worst_price[Ask] = mm->GetMbpAskPrice(mm->mbp.count - 1);

    if (!mm->symbol->IsRoll())
    {
        market_order_qty_[Bid] = touch_price[Bid] == 0.0 ? mm->GetMbpBidQty(0) : 0.0;
        market_order_qty_[Ask] = touch_price[Ask] == 0.0 ? mm->GetMbpAskQty(0) : 0.0;

        for (; touch_index[Bid] < mm->mbp.count; touch_index[Bid]++)
        {
            touch_price[Bid] = mm->GetMbpBidPrice(touch_index[Bid]);
            if (touch_price[Bid] > 0.0)
            {
                break;
            }
        }
        if (touch_index[Bid] == mm->mbp.count)
        {
            worst_index[Bid] = touch_index[Bid];
            touch_price[Bid] = INVALID_BID_PRICE;
            worst_price[Bid] = INVALID_BID_PRICE;
        }
        else
        {
            for (; worst_index[Bid] > touch_index[Bid]; worst_index[Bid]--)
            {
                worst_price[Bid] = mm->GetMbpBidPrice(worst_index[Bid] - 1);
                if (worst_price[Bid] > 0.0)
                {
                    break;
                }
            }
            if (worst_index[Bid] == touch_index[Bid])
            {
                worst_price[Bid] = touch_price[Bid];
            }
        }

        for (; touch_index[Ask] < mm->mbp.count; touch_index[Ask]++)
        {
            touch_price[Ask] = mm->GetMbpAskPrice(touch_index[Ask]);
            if (touch_price[Ask] > 0.0)
            {
                break;
            }
        }
        if (touch_index[Ask] == mm->mbp.count)
        {
            worst_index[Ask] = touch_index[Ask];
            touch_price[Ask] = INVALID_ASK_PRICE;
            worst_price[Ask] = INVALID_ASK_PRICE;
        }
        else
        {
            for (; worst_index[Ask] > touch_index[Ask]; worst_index[Ask]--)
            {
                worst_price[Ask] = mm->GetMbpAskPrice(worst_index[Ask] - 1);
                if (worst_price[Ask] > 0.0)
                {
                    break;
                }
            }
            if (worst_index[Ask] == touch_index[Ask])
            {
                worst_price[Ask] = touch_price[Ask];
            }
        }
    }
}

void MarketByOrderBook::CleanLevelsBetweenMarketByPriceGap(const MarketDataMessage *mm)
{
    message_delete_->ConstructBookMessageFromMarketMessage(mm);

    SPDLOG_INFO("am i here?11");
    Level *    tmp{nullptr};
    const bool is_volatility_interruption{
        symbol_ != nullptr && symbol_->GetDataSourceID() == DataSourceID::TSE && is_trade_ &&
        last_trade_price_ != 0.0 && last_trade_qty_ == 0.0};  // 盤中價格波動干預措施：暫緩撮合
    if (BRANCH_LIKELY(!is_volatility_interruption))
    {
        // clean bid levels above best bid
        Level *bid{touch_bid_};
        while (bid != nullptr && bid->price_ > market_by_price_entry_.touch_price_[Bid])
        {
            tmp                        = bid->next_;
            message_delete_->mbo.side  = bid->side_;
            message_delete_->mbo.price = bid->price_;
            ClearOrders(bid);
            bid = tmp;
        }

        // clean ask levels below best ask
        Level *ask{touch_ask_};
        while (ask != nullptr && ask->price_ < market_by_price_entry_.touch_price_[Ask])
        {
            tmp                        = ask->next_;
            message_delete_->mbo.side  = ask->side_;
            message_delete_->mbo.price = ask->price_;
            ClearOrders(ask);
            ask = tmp;
        }
    }
}

void MarketByOrderBook::CleanLevelsNotAmongMarketByPrice(const MarketDataMessage *mm)
{
    message_delete_->ConstructBookMessageFromMarketMessage(mm);

    Level *tmp{nullptr};

    // clean bid levels above best bid
    Level *   bid{touch_bid_};
    BookNolv  bid_index{market_by_price_entry_.touch_index_[Bid]};
    BookPrice bid_price{market_by_price_entry_.touch_price_[Bid]};
    while (bid != nullptr && bid->price_ > market_by_price_entry_.worst_price_[Bid])
    {
        while (bid_index < mm->mbp.count &&
               bid->price_ < (bid_price = mm->GetMbpBidPrice(bid_index)))
        {
            ++bid_index;
        }

        if (bid_index == mm->mbp.count)
        {
            break;
        }

        tmp = bid->next_;
        if (bid->price_ != bid_price)
        {
            message_delete_->mbo.side  = bid->side_;
            message_delete_->mbo.price = bid->price_;
            ClearOrders(bid);
        }
        bid = tmp;
    }

    // clean bid levels above best ask
    Level *   ask{touch_ask_};
    BookNolv  ask_index{market_by_price_entry_.touch_index_[Ask]};
    BookPrice ask_price{market_by_price_entry_.touch_price_[Ask]};
    while (ask != nullptr && ask->price_ < market_by_price_entry_.worst_price_[Ask])
    {
        while (ask_index < mm->mbp.count &&
               ask->price_ > (ask_price = mm->GetMbpAskPrice(ask_index)))
        {
            ++ask_index;
        }

        if (ask_index >= mm->mbp.count)
        {
            break;
        }

        tmp = ask->next_;
        if (ask->price_ != ask_price)
        {
            message_delete_->mbo.side  = ask->side_;
            message_delete_->mbo.price = ask->price_;
            ClearOrders(ask);
        }
        ask = tmp;
    }
}

void MarketByOrderBook::OnSnapshot(const MarketDataMessage *mm)
{
    if (mm->type == DataSourceType::MarketByPrice)
    {
        ParseMarketByPriceEntry(mm);

        CleanLevelsBetweenMarketByPriceGap(mm);

        const auto &touch_index_bid{market_by_price_entry_.touch_index_[Bid]};
        const auto &worst_index_bid{market_by_price_entry_.worst_index_[Bid]};
        const auto &touch_index_ask{market_by_price_entry_.touch_index_[Ask]};
        const auto &worst_index_ask{market_by_price_entry_.worst_index_[Ask]};
        for (BookNolv l{0}; l < mm->mbp.count; ++l)
        {
            if (BRANCH_LIKELY(l >= touch_index_bid && l < worst_index_bid))
            {
                ParseSnapshotOnBid(mm, -1, mm->GetMbpBidPrice(l), mm->GetMbpBidQty(l), 1);
            }
            if (BRANCH_LIKELY(l >= touch_index_ask && l < worst_index_ask))
            {
                ParseSnapshotOnAsk(mm, -1, mm->GetMbpAskPrice(l), mm->GetMbpAskQty(l), 1);
            }
        }

        CleanLevelsNotAmongMarketByPrice(mm);

        last_boundary_bid_ =
            worst_index_bid > 0 ? mm->GetMbpBidPrice(worst_index_bid - 1) : INVALID_BID_PRICE;
        last_boundary_ask_ =
            worst_index_ask > 0 ? mm->GetMbpAskPrice(worst_index_ask - 1) : INVALID_ASK_PRICE;
    }
    else if (mm->type == DataSourceType::MarketByOrder)
    {
        if (mm->mbo.side == BID)
        {
            ParseSnapshotOnBid(mm, mm->mbo.order_id, mm->GetMboPrice(), mm->mbo.qty, 1);
        }
        else
        {
            ParseSnapshotOnAsk(mm, mm->mbo.order_id, mm->GetMboPrice(), mm->mbo.qty, 1);
        }
    }

    ++book_statistic_snapshots_total_;

    UncrossMarketByOrderBook(mm);
}

void MarketByOrderBook::OnPacketEnd(const MarketDataMessage *mm)
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

    is_trade_ = false;
}

void MarketByOrderBook::OnSparseStop()
{
    EmitSparseStop();
}

void MarketByOrderBook::Reset()
{
    ResetWithoutCounts();
    book_statistic_adds_total_              = 0;
    book_statistic_deletes_total_           = 0;
    book_statistic_modify_with_price_total_ = 0;
    book_statistic_modify_with_qty_total_   = 0;
    book_statistic_snapshots_total_         = 0;
    book_statistic_trades_total_            = 0;
    book_statistic_packet_ends_total_       = 0;
}

Timestamp MarketByOrderBook::GetTime(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        if (BRANCH_LIKELY(nolv < all_bids_.size()))
        {
            return all_bids_.find_by_order(all_bids_.size() - nolv - 1)->second->time_;
        }
    }
    else
    {
        if (BRANCH_LIKELY(nolv < all_asks_.size()))
        {
            return all_asks_.find_by_order(all_asks_.size() - nolv - 1)->second->time_;
        }
    }
    return Timestamp();  // shouldn't come here
}

BookNolv MarketByOrderBook::GetNumberOfLevels(const BookSide side) const
{
    return side == BID ? all_bids_.size() : all_asks_.size();
}

const Level *MarketByOrderBook::GetTouchLevel(const BookSide side) const
{
    return side == BID ? touch_bid_ : touch_ask_;
}

const Level *MarketByOrderBook::GetTouchLevel(BidType *) const
{
    return touch_bid_;
}

const Level *MarketByOrderBook::GetTouchLevel(AskType *) const
{
    return touch_ask_;
}

const Level *MarketByOrderBook::GetWorstLevel(const BookSide side) const
{
    return side == BID ? all_bids_.begin()->second : all_asks_.begin()->second;
}

const Order *MarketByOrderBook::GetOrder(const Level *level, const BookQty qty) const
{
    if (!level)
    {
        return nullptr;
    }

    // GetOrder searches for the newest order with this given qty in this specified level. If found,
    // GetOrder will return that order. If not, nullptr will be returned. Note that this is linear
    // time O(N) operation.
    Order *order{level->head_};
    size_t count{0};
    while (order != nullptr)
    {
        if (order->qty_ == qty)
        {
            return order;
        }

        order = order->prev_;

        ++count;
        if (count > 100)
        {
            break;
        }
    }

    return nullptr;
}

const Level *MarketByOrderBook::GetLevelFromPrice(const BookSide & side,
                                                  const BookPrice &price) const
{
    if (side == BID)
    {
        auto it{all_bids_.find(price)};
        if (BRANCH_LIKELY(it != all_bids_.end()))
        {
            return it->second;
        }
    }
    else
    {
        auto it{all_asks_.find(price)};
        if (BRANCH_LIKELY(it != all_asks_.end()))
        {
            return it->second;
        }
    }
    return nullptr;
}

const Level *MarketByOrderBook::GetLevelFromPrice(const BookSide &side, const BookPrice &price,
                                                  BookNolv &nolv) const
{
    if (side == BID)
    {
        auto it{all_bids_.find(price)};
        if (BRANCH_LIKELY(it != all_bids_.end()))
        {
            nolv = all_bids_.size() - all_bids_.order_of_key(price) - 1;
            return it->second;
        }
    }
    else
    {
        auto it{all_asks_.find(price)};
        if (BRANCH_LIKELY(it != all_asks_.end()))
        {
            nolv = all_asks_.size() - all_asks_.order_of_key(price) - 1;
            return it->second;
        }
    }
    return nullptr;
}

// get methods for statistics
Timestamp MarketByOrderBook::GetLastTradeTime() const
{
    return last_trade_time_;
}

BookPrice MarketByOrderBook::GetLastTradePrice() const
{
    return last_trade_price_;
}

BookQty MarketByOrderBook::GetLastTradeQty() const
{
    return last_trade_qty_;
}

BookSide MarketByOrderBook::GetLastTradeSide() const
{
    return last_trade_side_;
}

void MarketByOrderBook::ResetOnlyBook()
{
    // cleaning orderbook
    std::vector<Level *> all_bid_levels;
    std::vector<Level *> all_ask_levels;
    for (auto x{all_bids_.begin()}; x != all_bids_.end(); ++x)
    {
        all_bid_levels.push_back(x->second);
    }
    for (auto x{all_asks_.begin()}; x != all_asks_.end(); ++x)
    {
        all_ask_levels.push_back(x->second);
    }

    message_delete_->mbo.side = BID;
    for (auto &x : all_bid_levels)
    {
        message_delete_->mbo.price = x->price_;
        ClearOrders(x);
    }
    message_delete_->mbo.side = ASK;
    for (auto &x : all_ask_levels)
    {
        message_delete_->mbo.price = x->price_;
        ClearOrders(x);
    }

    if (BRANCH_UNLIKELY(!all_bids_.empty()))  // all_bids_ should be empty now
    {
        abort();
    }
    if (BRANCH_UNLIKELY(!all_asks_.empty()))  // all_asks_ should be empty now
    {
        abort();
    }
}

void MarketByOrderBook::ResetWithoutCounts()
{
    ResetOnlyBook();
    // resetting counts in MarketByOrderBook
    touch_bid_       = nullptr;
    touch_ask_       = nullptr;
    packet_end_sent_ = false;
}

void MarketByOrderBook::Dump() const
{
    std::cout << "  DUMP BID ";
    for (auto i{all_bids_.rbegin()}; i != all_bids_.rend(); ++i)
    {
        std::cout << " " << i->first << "[" << i->second->qty_ << "] ";
    }
    std::cout << '\n';

    std::cout << "  DUMP ASK ";
    for (auto i{all_asks_.rbegin()}; i != all_asks_.rend(); ++i)
    {
        std::cout << " " << i->first << "[" << i->second->qty_ << "] ";
    }
    std::cout << '\n';
}

void MarketByOrderBook::DumpPrettyBook(size_t window) const
{
    std::cout << DumpString(window) << std::endl;
}

void MarketByOrderBook::DumpSnapshot(size_t window) const
{
    const size_t count{market_data_message_->mbp.count};

    std::cout << last_event_loop_time_ << '\n';
    printf("%16s        [%8s]        (%8s)\n", "PRICE", "QTY", "NORD");
    printf("%s", fmt::format("{:=>{}}\n", "", 64).c_str());
    for (size_t l{0}; l < count && l < window; ++l)
    {
        auto price{market_data_message_->GetMbpAskPrice(count - l - 1)};
        auto qty{market_data_message_->GetMbpAskQty(count - l - 1)};
        printf("%16.2f        [%8.2f]        (%8d)\n", price, qty, 1);
    }
    printf("%s", fmt::format("{:->{}}\n", "", 64).c_str());
    for (size_t l{0}; l < count && l < window; ++l)
    {
        auto price{market_data_message_->GetMbpBidPrice(l)};
        auto qty{market_data_message_->GetMbpBidQty(l)};
        printf("%16.2f        [%8.2f]        (%8d)\n", price, qty, 1);
    }
    printf("%s", fmt::format("{:=>{}}\n", "", 64).c_str());
}

const std::string MarketByOrderBook::DumpString(size_t window) const
{
    std::stringstream ss;

    size_t                                                        now_window{0};
    std::vector<std::tuple<BookPrice, BookQty, BookNord, double>> bids;
    std::vector<std::tuple<BookPrice, BookQty, BookNord, double>> asks;

    now_window = 0;
    for (auto it{all_asks_.rbegin()}; it != all_asks_.rend(); ++it)
    {
        if (now_window >= window)
        {
            break;
        }

        double age{0.0};
        double qty{0.0};

        const Order *order{it->second->head_};
        while (order != nullptr)
        {
            age += order->qty_ * (last_event_loop_time_ - order->time_).to_double();
            qty += order->qty_;
            order = order->prev_;
        }

        age = age / qty;

        asks.push_back(std::make_tuple(it->first, it->second->qty_, it->second->nord_, age));

        ++now_window;
    }

    now_window = 0;
    for (auto it{all_bids_.rbegin()}; it != all_bids_.rend(); ++it)
    {
        if (now_window >= window)
        {
            break;
        }

        double age{0.0};
        double qty{0.0};

        const Order *order{it->second->head_};
        while (order != nullptr)
        {
            age += order->qty_ * (last_event_loop_time_ - order->time_).to_double();
            qty += order->qty_;
            order = order->prev_;
        }

        age = age / qty;

        bids.push_back(std::make_tuple(it->first, it->second->qty_, it->second->nord_, age));

        ++now_window;
    }

    ss << last_event_loop_time_ << "\n";
    ss << "* symbol=[" << (GetSymbol() ? GetSymbol()->to_string() : "nullptr") << "]\n";
    ss << fmt::format(fmt::fg(fmt::color::gray), "{:=>{}}\n", "", 80);
    ss << fmt::format("{:>16s}{:8s}[{:>8s}]{:8s}({:>8s}){:8s}{{{:>16s}}}\n", "PRICE", "", "QTY", "",
                      "NORD", "", "AGE");
    ss << fmt::format(fmt::fg(fmt::color::gray), "{:=>{}}\n", "", 80);
    for (auto it{asks.crbegin()}; it != asks.crend(); ++it)
    {
        auto [price, qty, nord, age] = *it;
        ss << fmt::format("{}{:8s}[{:>8.2f}]{:8s}({:>8d}){:8s}{{{:>15.2f}s}}\n",
                          fmt::format(fmt::fg(fmt::terminal_color::green), "{:>16.2f}", price), "",
                          qty, "", nord, "", age);
    }
    ss << fmt::format(fmt::fg(fmt::color::gray), "{:->{}}\n", "", 80);
    for (auto it{bids.cbegin()}; it != bids.cend(); ++it)
    {
        auto [price, qty, nord, age] = *it;
        ss << fmt::format("{}{:8s}[{:>8.2f}]{:8s}({:>8d}){:8s}{{{:>15.2f}s}}\n",
                          fmt::format(fmt::fg(fmt::terminal_color::red), "{:>16.2f}", price), "",
                          qty, "", nord, "", age);
    }
    ss << fmt::format(fmt::fg(fmt::color::gray), "{:=>{}}\n", "", 80);

    ss << "*";
    ss << " last_trade_price=" << GetLastTradePrice();
    ss << " last_trade_qty=" << GetLastTradeQty();
    ss << "\n";

    ss << "*";
    ss << " ADDs=" << book_statistic_adds_total_;
    ss << " DELs=" << book_statistic_deletes_total_;
    ss << " MWPs=" << book_statistic_modify_with_price_total_;
    ss << " MWQs=" << book_statistic_modify_with_qty_total_;
    ss << " SNSs=" << book_statistic_snapshots_total_;
    ss << " TRDs=" << book_statistic_trades_total_;
    ss << " PKEs=" << book_statistic_packet_ends_total_;
    ss << "\n";

    return ss.str();
}

const std::string MarketByOrderBook::DumpLevelOrder(BookSide side, BookNolv nolv) const
{
    std::stringstream ss;

    size_t now_window{0};
    for (auto it{all_asks_.rbegin()}; it != all_asks_.rend(); ++it)
    {
        if (now_window == nolv)
        {
            ss << last_event_loop_time_ << " [" << it->first << "]";
            const Order *order{it->second->tail_};
            while (order != nullptr)
            {
                ss << " {" << order->qty_ << "}";
                order = order->next_;
            }
            ss << '\n';

            break;
        }

        ++now_window;
    }

    return ss.str();
}

bool MarketByOrderBook::IsValid() const
{
    return IsValidBid() && IsValidAsk() && touch_ask_->price_ > touch_bid_->price_;
}

bool MarketByOrderBook::IsValidBid() const
{
    if (symbol_ != nullptr && !symbol_->IsRoll())
    {
        return all_bids_.size() > 0 && touch_bid_->price_ > 0.0;
    }
    else
    {
        return all_bids_.size() > 0 && touch_bid_->price_ > INVALID_BID_PRICE;
    }
}

bool MarketByOrderBook::IsValidAsk() const
{
    return all_asks_.size() > 0 && touch_ask_->price_ < INVALID_ASK_PRICE;
}


bool MarketByOrderBook::IsFlip() const
{
    return prev_touch_price_[Bid] != touch_bid_->price_ ||
           prev_touch_price_[Ask] != touch_ask_->price_;
}

bool MarketByOrderBook::IsFlipUp() const
{
    return touch_bid_->price_ >= prev_touch_price_[Ask];
}

bool MarketByOrderBook::IsFlipDown() const
{
    return touch_ask_->price_ <= prev_touch_price_[Bid];
}

void MarketByOrderBook::CreateOrder(Level *level, Order *order)
{
    order->level_ = level;

    if (level->head_ != nullptr)
    {
        order->prev_        = level->head_;
        level->head_->next_ = order;
    }
    level->head_ = order;

    level->qty_ += order->qty_;
    level->nord_ += 1;

    message_add_->mbo.qty             = order->qty_;
    message_add_->mbo.nord            = 1;
    message_add_->order_              = order;
    message_add_->is_new_level_added_ = false;

    EmitPostAdd();
}

void MarketByOrderBook::CreateLevel(Level *level, Order *order)
{
    order->level_ = level;

    if (level->side_ == BID)
    {
        if (BRANCH_UNLIKELY((touch_bid_ == nullptr)))  // add at new touch
        {
            touch_bid_ = level;
        }
        else  // insert level
        {
            touch_bid_->prev_ = level;
            level->next_      = touch_bid_;
            touch_bid_        = level;
        }
    }
    else
    {
        if (BRANCH_UNLIKELY((touch_ask_ == nullptr)))  // add at new touch
        {
            touch_ask_ = level;
        }
        else  // insert level
        {
            touch_ask_->prev_ = level;
            level->next_      = touch_ask_;
            touch_ask_        = level;
        }
    }

    message_add_->mbo.qty             = order->qty_;
    message_add_->mbo.nord            = 1;
    message_add_->order_              = order;
    message_add_->is_new_level_added_ = true;

    if (is_overrunning_checking_)
    {
        message_add_->is_overrunning = false;

        const auto &side{message_add_->GetMarketByOrderSide()};
        const auto &price{message_add_->GetMarketByOrderPrice()};
        if (BRANCH_UNLIKELY(side == BID && price < last_boundary_bid_))
        {
            message_add_->is_overrunning = true;
        }
        else if (BRANCH_UNLIKELY(side == ASK && price > last_boundary_ask_))
        {
            message_add_->is_overrunning = true;
        }
    }

    EmitPostAdd();
}

void MarketByOrderBook::CreateLevel(Level *level, Order *order, Level *insert_l)
{
    order->level_ = level;

    if (level->side_ == BID)
    {
        if (BRANCH_UNLIKELY((touch_bid_ == nullptr)))  // add at touch
        {
            touch_bid_ = level;
        }
        else  // insert level
        {
            Level *tmp{insert_l->next_};
            insert_l->next_ = level;
            level->prev_    = insert_l;
            if (tmp)
            {
                tmp->prev_   = level;
                level->next_ = tmp;
            }
        }
    }
    else
    {
        if (BRANCH_UNLIKELY((touch_ask_ == nullptr)))  // add at touch
        {
            touch_ask_ = level;
        }
        else  // insert level
        {
            Level *tmp{insert_l->next_};
            insert_l->next_ = level;
            level->prev_    = insert_l;
            if (tmp)
            {
                tmp->prev_   = level;
                level->next_ = tmp;
            }
        }
    }

    message_add_->mbo.qty             = order->qty_;
    message_add_->mbo.nord            = 1;
    message_add_->order_              = order;
    message_add_->is_new_level_added_ = true;

    if (is_overrunning_checking_)
    {
        message_add_->is_overrunning = false;

        const auto &side{message_add_->GetMarketByOrderSide()};
        const auto &price{message_add_->GetMarketByOrderPrice()};
        if (BRANCH_UNLIKELY(side == BID && price < last_boundary_bid_))
        {
            message_add_->is_overrunning = true;
        }
        else if (BRANCH_UNLIKELY(side == ASK && price > last_boundary_ask_))
        {
            message_add_->is_overrunning = true;
        }
    }

    EmitPostAdd();
}

void MarketByOrderBook::RemoveOrder(Level *level, Order *order, bool operated)
{
    if (is_queue_like_)
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
    }
    else
    {
        if (order->prev_)
        {
            order->prev_->next_ = order->next_;
        }

        if (order->next_)
        {
            order->next_->prev_ = order->prev_;
        }

        if (order == level->head_)
        {
            level->head_ = order->prev_;
        }

        if (order == level->tail_)
        {
            level->tail_ = order->next_;
        }
    }

    level->qty_ -= order->qty_;
    level->nord_ -= 1;

    if (BRANCH_UNLIKELY(message_delete_->mbo.price != level->price_))
    {
        message_delete_->mbo.price = level->price_;
    }

    if (BRANCH_UNLIKELY(message_delete_->mbo.side != level->side_))
    {
        message_delete_->mbo.side = level->side_;
    }

    message_delete_->mbo.qty  = order->qty_;
    message_delete_->mbo.nord = 1;
    message_delete_->order_   = order;

    EmitPreDelete();

    order_factory_->DeleteOrder(order);
    if (level->nord_ == 0 || level->qty_ == 0.)
    {
        RemoveLevel(level);
    }

    EmitPostDelete();
}

void MarketByOrderBook::DecreaseOrderQty(Level *level, Order *order, BookQty qty)
{
    order->qty_ -= qty;
    level->qty_ -= qty;

    message_delete_->mbo.qty                 = qty;
    message_delete_->mbo.nord                = 1;
    message_delete_->order_                  = order;
    message_delete_->is_level_fully_deleted_ = false;

    EmitPreDelete();

    EmitPostDelete();
}

void MarketByOrderBook::IncreaseOrderQty(Level *level, Order *order, BookQty qty)
{
    order->qty_ += qty;
    level->qty_ += qty;

    message_add_->mbo.qty             = qty;
    message_add_->mbo.nord            = 1;
    message_add_->order_              = order;
    message_add_->is_new_level_added_ = false;

    EmitPostAdd();
}

void MarketByOrderBook::RemoveQty(Level *level, const BookQty qty, bool operated)
{
    if (qty >= level->qty_)
    {
        ClearOrders(level);
        return;
    }

    // search for the newest (not is_queue_like, i.e., is_stack_like) / oldest (is_queue_like)
    // order of this qty. if found, delete it. if not, delete the equivalent qty from the top
    Order *order{nullptr};
    if (!operated)
    {
        if (is_queue_like_)
        {
            order = level->tail_;
            while (order != nullptr)
            {
                if (order->qty_ == qty)
                {
                    RemoveOrder(level, order);
                    return;
                }
                order = order->next_;
            }
        }
        else
        {
            order = level->head_;
            while (order != nullptr)
            {
                if (order->qty_ == qty)
                {
                    RemoveOrder(level, order);
                    return;
                }
                order = order->prev_;
            }
        }
    }
    order = is_queue_like_ or operated ? level->tail_ : level->head_;

    Order * tmp{nullptr};
    BookQty qty_left_to_delete{qty};
    if (is_queue_like_ or operated)
    {
        while (order != nullptr && qty_left_to_delete != 0)
        {
            tmp   = order;
            order = order->next_;
            if (tmp->qty_ <= qty_left_to_delete)
            {
                RemoveOrder(level, tmp);
                qty_left_to_delete -= tmp->qty_;
            }
            else
            {
                DecreaseOrderQty(level, tmp, qty_left_to_delete);
                qty_left_to_delete -= qty_left_to_delete;
            }
        }
    }
    else
    {
        while (order != nullptr && qty_left_to_delete != 0)
        {
            tmp   = order;
            order = order->prev_;
            if (tmp->qty_ <= qty_left_to_delete)
            {
                RemoveOrder(level, tmp);
                qty_left_to_delete -= tmp->qty_;
            }
            else
            {
                DecreaseOrderQty(level, tmp, qty_left_to_delete);
                qty_left_to_delete -= qty_left_to_delete;
            }
        }
    }
}

void MarketByOrderBook::ClearOrders(Level *level)
{
    Order *order{level->head_};
    Order *tmp{nullptr};

    message_delete_->is_level_fully_deleted_ = true;
    while (order != nullptr)
    {
        tmp   = order;
        order = order->prev_;
        RemoveOrder(level, tmp);
    }
    message_delete_->is_level_fully_deleted_ = false;
}

void MarketByOrderBook::RemoveLevel(Level *level)
{
    if (level->prev_)
    {
        level->prev_->next_ = level->next_;
    }

    if (level->next_)
    {
        level->next_->prev_ = level->prev_;
    }

    if (level->side_ == BID)
    {
        if (touch_bid_ == level)
        {
            touch_bid_ = (level->next_ ? level->next_ : nullptr);
        }
        all_bids_.erase(level->price_);
    }
    else
    {
        if (touch_ask_ == level)
        {
            touch_ask_ = (level->next_ ? level->next_ : nullptr);
        }
        all_asks_.erase(level->price_);
    }

    level_factory_->RemoveLevel(level);
}

void MarketByOrderBook::EmitPreDelete() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    if (is_rejoining_checking_)
    {
        message_delete_->is_rejoining = false;

        const auto &side{message_delete_->GetMarketByOrderSide()};
        const auto &price{message_delete_->GetMarketByOrderPrice()};
        if (BRANCH_UNLIKELY(side == BID && price < last_boundary_bid_))
        {
            message_delete_->is_rejoining = true;
        }
        else if (BRANCH_UNLIKELY(side == ASK && price > last_boundary_ask_))
        {
            message_delete_->is_rejoining = true;
        }
    }

    for (auto &listener : listeners_pre_)
    {
        listener->OnPreBookDelete(last_event_loop_time_, message_delete_);
    }
}

void MarketByOrderBook::EmitPostAdd() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    if (is_rejoining_checking_)
    {
        message_add_->is_rejoining = false;

        const auto &side{message_add_->GetMarketByOrderSide()};
        const auto &price{message_add_->GetMarketByOrderPrice()};
        if (BRANCH_UNLIKELY(side == BID && price < last_boundary_bid_))
        {
            message_add_->is_rejoining = true;
        }
        else if (BRANCH_UNLIKELY(side == ASK && price > last_boundary_ask_))
        {
            message_add_->is_rejoining = true;
        }
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookAdd(last_event_loop_time_, message_add_);
    }
}

void MarketByOrderBook::EmitPostDelete() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    if (is_rejoining_checking_)
    {
        message_delete_->is_rejoining = false;

        const auto &side{message_delete_->GetMarketByOrderSide()};
        const auto &price{message_delete_->GetMarketByOrderPrice()};
        if (BRANCH_UNLIKELY(side == BID && price < last_boundary_bid_))
        {
            message_delete_->is_rejoining = true;
        }
        else if (BRANCH_UNLIKELY(side == ASK && price > last_boundary_ask_))
        {
            message_delete_->is_rejoining = true;
        }
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookDelete(last_event_loop_time_, message_delete_);
    }
}

void MarketByOrderBook::UncrossMarketByOrderBook(const MarketDataMessage *mm)
{
    message_delete_->ConstructBookMessageFromMarketMessage(mm);

    // clean the order book by removing crossed orders which are older
    bool is_crossed{false};
    bool is_locked{false};

    Level *bid{touch_bid_};
    Level *ask{touch_ask_};
    Level *tmp{nullptr};
    while (bid != nullptr && ask != nullptr &&
           ((is_crossed = ask->price_ < bid->price_) or (is_locked = ask->price_ == bid->price_)))
    {
        if (bid->time_ < ask->time_)
        {
            tmp                        = bid->next_;
            message_delete_->mbo.side  = bid->side_;
            message_delete_->mbo.price = bid->price_;
            ClearOrders(bid);
            bid = tmp;
        }
        else
        {
            tmp                        = ask->next_;
            message_delete_->mbo.side  = ask->side_;
            message_delete_->mbo.price = ask->price_;
            ClearOrders(ask);
            ask = tmp;
        }

        if (is_crossed)
        {
            ++book_statistic_uncross_book_total_;
        }

        if (is_locked)
        {
            ++book_statistic_unlock_book_total_;
        }
    }
}

void MarketByOrderBook::CleanMarketByOrderBookAfterTrade(const BookSide           side,
                                                         const MarketDataMessage *mm)
{
    if (side == BID)
    {
        while (touch_ask_ != nullptr && mm->GetTradePrice() > touch_ask_->price_)
        {
            message_delete_->ConstructBookMessageFromMarketMessage(mm);
            message_delete_->mbo.price = touch_ask_->price_;
            message_delete_->mbo.side  = ASK;
            ClearOrders(touch_ask_);
            ++book_statistic_heuristic_deletes_after_trades_total_;
        }
    }
    else
    {
        while (touch_bid_ != nullptr && mm->GetTradePrice() < touch_bid_->price_)
        {
            message_delete_->ConstructBookMessageFromMarketMessage(mm);
            message_delete_->mbo.price = touch_bid_->price_;
            message_delete_->mbo.side  = BID;
            ClearOrders(touch_bid_);
            ++book_statistic_heuristic_deletes_after_trades_total_;
        }
    }
}

}  // namespace alphaone
