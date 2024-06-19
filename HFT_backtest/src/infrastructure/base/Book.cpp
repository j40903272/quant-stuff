#include "Book.h"

#include "infrastructure/base/BookDataListener.h"
#include "infrastructure/common/util/Logger.h"

namespace alphaone
{
Book::Book(const Symbol *symbol, const bool is_emit_allowed_only_when_valid)
    : market_data_message_{nullptr}
    , symbol_{symbol}
    , is_emit_allowed_only_when_valid_{is_emit_allowed_only_when_valid}
    , last_event_loop_time_{Timestamp::invalid()}
    , last_event_time_{Timestamp::invalid(), Timestamp::invalid()}
    , last_warned_time_{Timestamp::invalid(), Timestamp::invalid()}
    , book_statistic_adds_total_{0}
    , book_statistic_deletes_total_{0}
    , book_statistic_modify_with_price_total_{0}
    , book_statistic_modify_with_qty_total_{0}
    , book_statistic_snapshots_total_{0}
    , book_statistic_trades_total_{0}
    , book_statistic_packet_ends_total_{0}
    , book_statistic_heuristic_deletes_after_trades_total_{0}
    , message_add_{new BookDataMessageAdd{}}
    , message_delete_{new BookDataMessageDelete{}}
    , message_modify_with_price_{new BookDataMessageModifyWithPrice{}}
    , message_modify_with_qty_{new BookDataMessageModifyWithQty{}}
    , message_snapshot_{new BookDataMessageSnapshot{}}
    , message_implied_order_{new BookDataMessage{}}
    , message_trade_{new BookDataMessageTrade{}}
    , message_packet_end_{new BookDataMessagePacketEnd{}}
    , message_sparse_stop_{new BookDataMessageSparseStop{}}
    , packet_end_sent_{false}
    , prev_touch_price_{-1.0, -1.0}
{
    std::memset(&message_implied_order_->implied, 0, sizeof(message_implied_order_->implied));
    message_add_->market_data_message_type               = MarketDataMessageType_Add;
    message_delete_->market_data_message_type            = MarketDataMessageType_Delete;
    message_modify_with_price_->market_data_message_type = MarketDataMessageType_ModifyWithPrice;
    message_modify_with_qty_->market_data_message_type   = MarketDataMessageType_ModifyWithQty;
    message_snapshot_->market_data_message_type          = MarketDataMessageType_Snapshot;
    message_trade_->market_data_message_type             = MarketDataMessageType_Trade;
    message_packet_end_->market_data_message_type        = MarketDataMessageType_PacketEnd;
}

Book::~Book()
{
    delete message_sparse_stop_;
    delete message_packet_end_;
    delete message_trade_;
    delete message_implied_order_;
    delete message_snapshot_;
    delete message_modify_with_qty_;
    delete message_modify_with_price_;
    delete message_delete_;
    delete message_add_;
}

BookPrice Book::GetLastTouchPrice(const BookSide side) const
{
    return prev_touch_price_[side];
}

BookPrice Book::GetImpliedOrderPrice(const BookSide side) const
{
    if (side == BID)
    {
        return message_implied_order_->implied.bid_price[0] == 0.0
                   ? INVALID_BID_PRICE
                   : message_implied_order_->implied.bid_price[0] / symbol_->GetDecimalConverter();
    }
    else
    {
        return message_implied_order_->implied.ask_price[0] == 0.0
                   ? INVALID_ASK_PRICE
                   : message_implied_order_->implied.ask_price[0] / symbol_->GetDecimalConverter();
    }
}

BookQty Book::GetImpliedOrderQty(const BookSide side) const
{
    if (side == BID)
    {
        return message_implied_order_->implied.bid_qty[0];
    }
    else
    {
        return message_implied_order_->implied.ask_qty[0];
    }
}

const Symbol *Book::GetSymbol() const
{
    return symbol_;
}

void Book::OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet)
{
    last_event_loop_time_ = mm->provider_time;  // TODO: Is this a suitable choice here?
    market_data_message_  = mm;
    raw_packet_           = raw_packet;

    switch (mm->market_data_message_type)
    {
    case MarketDataMessageType_Add:
        OnAdd(mm);
        last_event_time_[BookPart]                  = last_event_loop_time_;
        packet_end_sent_                            = false;
        message_packet_end_->exchange_time          = mm->provider_time;
        message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        if (mm->mbo.is_packet_end)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_Delete:
        OnDelete(mm);
        last_event_time_[BookPart]                  = last_event_loop_time_;
        packet_end_sent_                            = false;
        message_packet_end_->exchange_time          = mm->provider_time;
        message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        if (mm->mbo.is_packet_end)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_ModifyWithPrice:
        OnModifyWithPrice(mm);
        last_event_time_[BookPart]                  = last_event_loop_time_;
        packet_end_sent_                            = false;
        message_packet_end_->exchange_time          = mm->provider_time;
        message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        if (mm->mbo.is_packet_end)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_ModifyWithQty:
        OnModifyWithQty(mm);
        last_event_time_[BookPart]                  = last_event_loop_time_;
        packet_end_sent_                            = false;
        message_packet_end_->exchange_time          = mm->provider_time;
        message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        if (mm->mbo.is_packet_end)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_Snapshot:
        OnSnapshot(mm);
        last_event_time_[BookPart]                  = last_event_loop_time_;
        packet_end_sent_                            = false;
        message_packet_end_->exchange_time          = mm->provider_time;
        message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        if (mm->mbp.is_packet_end && mm->type == DataSourceType::MarketByPrice)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_Implied:
        message_implied_order_->ConstructBookMessageFromMarketMessage(mm);
        break;

    case MarketDataMessageType_Trade:
        if (BRANCH_LIKELY(mm->trade.is_not_duplicate_))
        {
            OnTrade(mm);
            last_event_time_[TradePart]                 = last_event_loop_time_;
            packet_end_sent_                            = false;
            message_packet_end_->exchange_time          = mm->provider_time;
            message_packet_end_->last_market_data_type_ = mm->market_data_message_type;
        }
        if (mm->trade.is_packet_end)
        {
            OnPacketEnd(mm);
            packet_end_sent_ = true;
        }
        break;

    case MarketDataMessageType_PacketEnd:
        if (packet_end_sent_)
        {
            break;
        }
        OnPacketEnd(mm);
        packet_end_sent_ = true;
        break;

    default:
        break;
    }
}

void Book::AddPreBookListener(BookDataListener *listener)
{
    if (listener == nullptr)
    {
        return;
    }

    if (std::find(listeners_pre_.begin(), listeners_pre_.end(), listener) == listeners_pre_.end())
    {
        listeners_pre_.push_back(listener);
    }
}

void Book::AddPostBookListener(BookDataListener *listener)
{
    if (listener == nullptr)
    {
        return;
    }

    if (std::find(listeners_post_.begin(), listeners_post_.end(), listener) ==
        listeners_post_.end())
    {
        listeners_post_.push_back(listener);
    }
}

void Book::EmitPreDelete() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_pre_)
    {
        listener->OnPreBookDelete(last_event_loop_time_, message_delete_);
    }
}

void Book::EmitPostAdd() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookAdd(last_event_loop_time_, message_add_);
    }
}

void Book::EmitPostDelete() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookDelete(last_event_loop_time_, message_delete_);
    }
}

void Book::EmitPostModifyWithPrice() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookModifyWithPrice(last_event_loop_time_, message_modify_with_price_);
    }
}

void Book::EmitPostModifyWithQty() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookModifyWithQty(last_event_loop_time_, message_modify_with_qty_);
    }
}

void Book::EmitPostSnapshot() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPostBookSnapshot(last_event_loop_time_, message_snapshot_);
    }
}

void Book::EmitPostTrade() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnTrade(last_event_loop_time_, message_trade_);
    }
}

void Book::EmitPacketEnd() const
{
    if (BRANCH_UNLIKELY(is_emit_allowed_only_when_valid_ && !IsValid()))
    {
        return;
    }

    for (auto &listener : listeners_post_)
    {
        listener->OnPacketEnd(last_event_loop_time_, message_packet_end_);
    }
}

void Book::EmitSparseStop() const
{
    for (auto &listener : listeners_post_)
    {
        listener->OnSparseStop(message_sparse_stop_->event_loop_time_, message_sparse_stop_);
    }
}
}  // namespace alphaone
