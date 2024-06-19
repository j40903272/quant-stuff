#ifndef BOOK_H
#define BOOK_H

#include "infrastructure/base/BookDataSource.h"
#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/BookDataMessage.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/EnumToString.h"

#include <vector>


namespace alphaone
{

struct BookCheckInfo
{
    BookCheckInfo()                = default;
    BookCheckInfo(BookCheckInfo &) = default;
    BookCheckInfo(const Duration &sparse_book, const Duration &sparse_trade,
                  const Duration &warning_book, const Duration &warning_trade)
        : sparse_duration_{sparse_book, sparse_trade}
        , warning_duration_{warning_book, warning_trade}
    {
    }
    // constructor from seconds
    BookCheckInfo(const double sparse_book, const double sparse_trade, const double warning_book,
                  const double warning_trade)
        : sparse_duration_{Duration::from_sec(sparse_book), Duration::from_sec(sparse_trade)}
        , warning_duration_{Duration::from_sec(warning_book), Duration::from_sec(warning_trade)}
    {
    }
    Duration sparse_duration_[BookTradePart];
    Duration warning_duration_[BookTradePart];
};

class BookDataListener;

/* -------------------------------------------------------------------------------------------------
Book objects are supposed to be a MarketDataListener and a BookDataSource, i.e., they have
MarketData callbacks that will be called when MarketData events have been triggered. They will call
Book callbacks of BookDataListeners when Book events have been triggered.
------------------------------------------------------------------------------------------------- */
class Book : public BookDataSource, public MarketDataListener
{
  public:
    Book(const Symbol *symbol, const bool is_emit_allowed_only_when_valid);
    ~Book();

    // get book type -- need to be defined explicitly;
    virtual DataSourceType GetType() const = 0;

    // get methods for book information
    virtual BookPrice GetPrice(const BookSide side, const BookNolv nolv = 0) const = 0;
    virtual BookQty   GetQty(const BookSide side, const BookNolv nolv = 0) const   = 0;
    virtual BookNord  GetNord(const BookSide side, const BookNolv nolv = 0) const  = 0;

    // for taifex, this method always returns 5. Nolv means number of levels.
    virtual BookNolv GetNolv(const BookSide side) const = 0;

    // get methods for directional information -- utilized when using alpha models
    virtual BookPrice GetMidPrice() const      = 0;
    virtual BookPrice GetWeightedPrice() const = 0;
    virtual BookQty   GetMidQty() const        = 0;

    // get methods for gap information -- utilized when using gap models
    virtual BookPrice GetSpread(const BookNolv b_lvl = 0, const BookNolv a_lvl = 0) const = 0;
    virtual BookPrice GetSpreadAsPercentOfMid(const BookNolv b_lvl = 0,
                                              const BookNolv a_lvl = 0) const             = 0;
    virtual BookPrice GetSpreadAsPercentOfBid(const BookNolv b_lvl = 0,
                                              const BookNolv a_lvl = 0) const             = 0;
    virtual BookPrice GetSpreadAsReturn(const BookNolv b_lvl = 0,
                                        const BookNolv a_lvl = 0) const                   = 0;

    // get methods for complicated book information -- useful when designing market making strategy
    virtual BookPrice GetPriceBehindQty(const BookSide &side, const BookQty &qty) const     = 0;
    virtual BookPrice GetPriceBeforeQty(const BookSide &side, const BookQty &qty) const     = 0;
    virtual BookQty   GetQtyBehindPrice(const BookSide &side, const BookPrice &price) const = 0;

    // get methods for statistics
    virtual Timestamp GetLastTradeTime() const  = 0;
    virtual BookPrice GetLastTradePrice() const = 0;
    virtual BookQty   GetLastTradeQty() const   = 0;
    virtual BookPrice GetLastTouchPrice(const BookSide side) const;
    BookPrice         GetImpliedOrderPrice(const BookSide side) const;
    BookQty           GetImpliedOrderQty(const BookSide side) const;
    size_t            GetNumberAdds() const;
    size_t            GetNumberDeletes() const;
    size_t            GetNumberTrades() const;
    size_t            GetNumberSnapshots() const;
    size_t            GetNumberModifyWithPrices() const;
    size_t            GetNumberModifyWithQtys() const;
    size_t            GetNumberPacketEnds() const;
    Timestamp         GetLastEventLoopTime() const;
    Timestamp         GetLastBookEventLoopTime() const;
    Timestamp         GetLastTradeEventLoopTime() const;

    // public raw market data callbacks
    virtual void OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet);

    // public raw market data callbacks
    virtual void OnAdd(const MarketDataMessage *)             = 0;  // self
    virtual void OnDelete(const MarketDataMessage *)          = 0;  // self
    virtual void OnModifyWithPrice(const MarketDataMessage *) = 0;  // self
    virtual void OnModifyWithQty(const MarketDataMessage *)   = 0;  // self
    virtual void OnSnapshot(const MarketDataMessage *)        = 0;  // self
    virtual void OnTrade(const MarketDataMessage *)           = 0;  // self
    virtual void OnPacketEnd(const MarketDataMessage *)       = 0;  // self
    virtual void OnSparseStop()                               = 0;  // self

    // reset OrderBook instance if needed
    virtual void Reset() = 0;

    // add listener
    void AddPreBookListener(BookDataListener *listener);
    void AddPostBookListener(BookDataListener *listener);

    // book event check
    template <BookTrade event_type>
    void CheckEventTime(const Timestamp event_loop_time, const Timestamp call_back_time,
                        void *structure)
    {
        auto check_info = reinterpret_cast<BookCheckInfo *>(structure);
        if (last_event_time_[event_type].is_valid() &&
            last_event_time_[event_type] + check_info->sparse_duration_[event_type] <
                event_loop_time)
        {
            SPDLOG_ERROR("[Too Long No {}] {} last event time = {}, current_event_loop_time = {}, "
                         "checking_duration = {}",
                         EnumToString::ToString(event_type), (*symbol_),
                         last_event_time_[event_type], event_loop_time,
                         check_info->sparse_duration_[event_type]);
            if (!last_warned_time_[event_type].is_valid() ||
                last_warned_time_[event_type] + check_info->warning_duration_[event_type] <
                    event_loop_time)
            {
                // neeed to define the warning behavior here
                message_sparse_stop_->event_type       = event_type;
                message_sparse_stop_->event_loop_time_ = event_loop_time;
                message_sparse_stop_->last_event_time_ = last_event_time_[event_type];
                message_sparse_stop_->symbol_          = symbol_;
                OnSparseStop();
                last_warned_time_[event_type] = event_loop_time;
            }
        }
    }

    // round
    inline BookPrice RoundToNearest(const BookPrice price, const BookSide side,
                                    const bool improved = false) const
    {
        if (side == BID)
        {
            const BookPrice tick{symbol_->GetTickSize(price, false)};

            if (improved)
            {
                return std::floor(price / tick) * tick + tick;
            }
            else
            {
                return std::floor(price / tick) * tick;
            }
        }
        else
        {
            const BookPrice tick{symbol_->GetTickSize(price, true)};

            if (improved)
            {
                return std::ceil(price / tick) * tick - tick;
            }
            else
            {
                return std::ceil(price / tick) * tick;
            }
        }
    }

    // dump book objects for debugging (or print on telnet directly if we would like to do this)
    virtual void              Dump() const                        = 0;
    virtual const std::string DumpString(size_t window = 5) const = 0;
    virtual const std::string DumpDepths(const BookNolv nolv, const std::string &delimiter) const
    {
        std::ostringstream buffer;
        buffer.str("");

        for (BookNolv lcount{0}; lcount <= nolv; ++lcount)
        {
            // clang-format off
            buffer << delimiter << "bp_" << lcount << "=" << GetPrice(BID, lcount)
                   << delimiter << "bq_" << lcount << "=" << GetQty(BID, lcount)
                   << delimiter << "ap_" << lcount << "=" << GetPrice(ASK, lcount)
                   << delimiter << "aq_" << lcount << "=" << GetQty(ASK, lcount);
            // clang-format on
        }

        return buffer.str();
    }

    // check
    virtual bool IsValid() const    = 0;
    virtual bool IsValidBid() const = 0;
    virtual bool IsValidAsk() const = 0;
    virtual bool IsFlip() const     = 0;
    virtual bool IsFlipUp() const   = 0;
    virtual bool IsFlipDown() const = 0;
    /*
    Check:
    * There are bids
    * There are asks
    * Best ask price is larger than 0 if there are asks
    * Best ask price is higher than best bid price if there are bids and there are asks
    */

    const Symbol *GetSymbol() const;
    void *        GetRawPacket() const
    {
        return raw_packet_;
    }

  private:
    void *raw_packet_{nullptr};

  protected:
    const MarketDataMessage *market_data_message_;
    const Symbol *           symbol_;
    const bool               is_emit_allowed_only_when_valid_;

    // time statistics
    Timestamp last_event_loop_time_;
    Timestamp last_event_time_[BookTradePart];
    Timestamp last_warned_time_[BookTradePart];

    // book statistics
    size_t book_statistic_adds_total_;
    size_t book_statistic_deletes_total_;
    size_t book_statistic_modify_with_price_total_;
    size_t book_statistic_modify_with_qty_total_;
    size_t book_statistic_snapshots_total_;
    size_t book_statistic_trades_total_;
    size_t book_statistic_packet_ends_total_;
    size_t book_statistic_heuristic_deletes_after_trades_total_;

    // for the efficiency of sending orders back out when calling emit methods
    BookDataMessageAdd *            message_add_;
    BookDataMessageDelete *         message_delete_;
    BookDataMessageModifyWithPrice *message_modify_with_price_;
    BookDataMessageModifyWithQty *  message_modify_with_qty_;
    BookDataMessageSnapshot *       message_snapshot_;
    BookDataMessage *               message_implied_order_;
    BookDataMessageTrade *          message_trade_;
    BookDataMessagePacketEnd *      message_packet_end_;
    BookDataMessageSparseStop *     message_sparse_stop_;

    std::vector<BookDataListener *> listeners_pre_;

    virtual void EmitPreDelete() const;

    std::vector<BookDataListener *> listeners_post_;

    virtual void EmitPostAdd() const;
    virtual void EmitPostDelete() const;
    virtual void EmitPostModifyWithPrice() const;
    virtual void EmitPostModifyWithQty() const;
    virtual void EmitPostSnapshot() const;
    virtual void EmitPostTrade() const;
    virtual void EmitPacketEnd() const;
    virtual void EmitSparseStop() const;

    // status for packet end
    bool packet_end_sent_;

    // members for some check
    BookPrice prev_touch_price_[AskBid];
};

inline size_t Book::GetNumberAdds() const
{
    return book_statistic_adds_total_;
}

inline size_t Book::GetNumberDeletes() const
{
    return book_statistic_deletes_total_;
}

inline size_t Book::GetNumberTrades() const
{
    return book_statistic_trades_total_;
}

inline size_t Book::GetNumberSnapshots() const
{
    return book_statistic_snapshots_total_;
}

inline size_t Book::GetNumberModifyWithPrices() const
{
    return book_statistic_modify_with_price_total_;
}

inline size_t Book::GetNumberModifyWithQtys() const
{
    return book_statistic_modify_with_qty_total_;
}

inline size_t Book::GetNumberPacketEnds() const
{
    return book_statistic_packet_ends_total_;
}

inline Timestamp Book::GetLastEventLoopTime() const
{
    return last_event_loop_time_;
}

inline Timestamp Book::GetLastBookEventLoopTime() const
{
    return last_event_time_[BookPart];
}

inline Timestamp Book::GetLastTradeEventLoopTime() const
{
    return last_event_time_[TradePart];
}
}  // namespace alphaone
#endif
