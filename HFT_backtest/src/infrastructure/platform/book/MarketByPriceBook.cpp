#include "MarketByPriceBook.h"

#include "infrastructure/common/math/Math.h"

namespace alphaone
{
MarketByPriceBook::MarketByPriceBook(const Symbol *symbol,
                                     const bool    is_emit_allowed_only_when_valid)
    : Book{symbol, is_emit_allowed_only_when_valid}
    , market_data_message_market_by_price_time_{Timestamp::invalid()}
    , market_data_message_trade_time_{Timestamp::invalid()}
{
}

MarketByPriceBook::~MarketByPriceBook()
{
    listeners_pre_.clear();
    listeners_post_.clear();
    Reset();
}

// get methods for book information
BookPrice MarketByPriceBook::GetPrice(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        if (BRANCH_LIKELY(nolv < market_data_message_market_by_price_.count))
        {
            return market_data_message_market_by_price_.bid_price[nolv] /
                   symbol_->GetDecimalConverter();
        }
        else
        {
            return INVALID_BID_PRICE;
        }
    }
    else
    {
        if (BRANCH_LIKELY(nolv < market_data_message_market_by_price_.count))
        {
            return market_data_message_market_by_price_.ask_price[nolv] /
                   symbol_->GetDecimalConverter();
        }
        else
        {
            return INVALID_ASK_PRICE;
        }
    }
}

BookQty MarketByPriceBook::GetQty(const BookSide side, const BookNolv nolv) const
{
    if (side == BID)
    {
        return market_data_message_market_by_price_.bid_qty[nolv];
    }
    else
    {
        return market_data_message_market_by_price_.ask_qty[nolv];
    }
}

BookNord MarketByPriceBook::GetNord(const BookSide side, const BookNolv nolv) const
{
    return 1;
}

// for taifex, this method always returns 5. Nolv means number of levels.
BookNolv MarketByPriceBook::GetNolv(const BookSide side) const
{
    return market_data_message_market_by_price_.count;
}

// get methods for directional information -- utilized when using alpha models
BookPrice MarketByPriceBook::GetMidPrice() const
{
    return 0.5 *
           (market_data_message_market_by_price_.bid_price[0] +
            market_data_message_market_by_price_.ask_price[0]) /
           symbol_->GetDecimalConverter();
}

BookPrice MarketByPriceBook::GetWeightedPrice() const
{
    return (GetQty(ASK) * GetPrice(BID) + GetQty(BID) * GetPrice(ASK)) /
           (GetQty(BID) + GetQty(ASK));
}

BookQty MarketByPriceBook::GetMidQty() const
{
    return 0.5 * (market_data_message_market_by_price_.bid_qty[0] +
                  market_data_message_market_by_price_.ask_qty[0]);
}

// get methods for gap information -- utilized when using gap models
BookPrice MarketByPriceBook::GetSpread(const BookNolv b_lvl, const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(GetNolv(BID) < b_lvl or GetNolv(ASK) < a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }
    else
    {
        return (GetPrice(ASK, a_lvl) - GetPrice(BID, b_lvl));
    }
}

BookPrice MarketByPriceBook::GetSpreadAsPercentOfMid(const BookNolv b_lvl,
                                                     const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(GetNolv(BID) < b_lvl or GetNolv(ASK) < a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }
    else
    {
        return (GetPrice(ASK, a_lvl) - GetPrice(BID, b_lvl)) / GetMidPrice();
    }
}

BookPrice MarketByPriceBook::GetSpreadAsPercentOfBid(const BookNolv b_lvl,
                                                     const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(GetNolv(BID) < b_lvl or GetNolv(ASK) < a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }
    else
    {
        return (GetPrice(ASK, a_lvl) - GetPrice(BID, b_lvl)) / GetPrice(BID, b_lvl);
    }
}

BookPrice MarketByPriceBook::GetSpreadAsReturn(const BookNolv b_lvl, const BookNolv a_lvl) const
{
    if (BRANCH_UNLIKELY(GetNolv(BID) < b_lvl or GetNolv(ASK) < a_lvl))
    {
        return static_cast<BookPrice>(NaN);
    }
    else
    {
        return y_log(GetPrice(ASK, a_lvl) / GetPrice(BID, b_lvl));
    }
}

// get methods for complicated book information -- useful when designing market making strategy
BookPrice MarketByPriceBook::GetPriceBehindQty(const BookSide &side, const BookQty &qty) const
{
    return 0.0;
}

BookPrice MarketByPriceBook::GetPriceBeforeQty(const BookSide &side, const BookQty &qty) const
{
    return 0.0;
}

BookQty MarketByPriceBook::GetQtyBehindPrice(const BookSide &side, const BookPrice &price) const
{
    return 0.0;
}

// get methods for statistics
Timestamp MarketByPriceBook::GetLastTradeTime() const
{
    return market_data_message_trade_time_;
}

BookPrice MarketByPriceBook::GetLastTradePrice() const
{
    return market_data_message_trade_.price / symbol_->GetDecimalConverter();
}

BookQty MarketByPriceBook::GetLastTradeQty() const
{
    return market_data_message_trade_.qty;
}

// public raw market data callbacks
void MarketByPriceBook::OnAdd(const MarketDataMessage *mm)
{
}  // self

void MarketByPriceBook::OnDelete(const MarketDataMessage *mm)
{
}  // self

void MarketByPriceBook::OnModifyWithPrice(const MarketDataMessage *mm)
{
}  // self

void MarketByPriceBook::OnModifyWithQty(const MarketDataMessage *mm)
{
}  // self

void MarketByPriceBook::OnSnapshot(const MarketDataMessage *mm)
{
    message_snapshot_->ConstructBookMessageFromMarketMessage(mm);
    market_data_message_market_by_price_time_ = mm->provider_time;
    market_data_message_market_by_price_      = mm->mbp;
    EmitPostSnapshot();
}  // self

void MarketByPriceBook::OnTrade(const MarketDataMessage *mm)
{
    message_trade_->ConstructBookMessageFromMarketMessage(mm);
    market_data_message_trade_time_ = mm->provider_time;
    market_data_message_trade_      = mm->trade;
    EmitPostTrade();
}  // self

void MarketByPriceBook::OnPacketEnd(const MarketDataMessage *mm)
{
    message_packet_end_->ConstructBookMessageFromMarketMessage(mm);
    EmitPacketEnd();

    if (BRANCH_LIKELY(IsValid()))
    {
        prev_touch_price_[Ask] = GetPrice(ASK);
        prev_touch_price_[Bid] = GetPrice(BID);
    }
}  // self

void MarketByPriceBook::OnSparseStop()
{
    EmitSparseStop();
}

// reset MarketByPriceBook instance if needed
void MarketByPriceBook::Reset()
{
    book_statistic_adds_total_              = 0;
    book_statistic_deletes_total_           = 0;
    book_statistic_modify_with_price_total_ = 0;
    book_statistic_modify_with_qty_total_   = 0;
    book_statistic_snapshots_total_         = 0;
    book_statistic_trades_total_            = 0;
    book_statistic_packet_ends_total_       = 0;
}

// dump book objects for debugging (or print on telnet directly if we would like to do this)
void MarketByPriceBook::Dump() const
{
    std::cout << "  DUMP BID ";
    for (BookNolv i{0}; i < GetNolv(BID); ++i)
    {
        std::cout << " " << GetPrice(BID, i) << "[" << GetQty(BID, i) << "] ";
    }
    std::cout << '\n';

    std::cout << "  DUMP ASK ";
    for (BookNolv i{0}; i < GetNolv(ASK); ++i)
    {
        std::cout << " " << GetPrice(ASK, i) << "[" << GetQty(ASK, i) << "] ";
    }
    std::cout << '\n';
}

const std::string MarketByPriceBook::DumpString(size_t window) const
{
    std::stringstream ss;

    const size_t count{market_data_message_market_by_price_.count};

    ss << last_event_loop_time_ << '\n';

    ss << "* symbol=" << GetSymbol() << '\n';

    {
        char buffer[256];
        sprintf(buffer, "%16s        [%8s]        (%8s)\n", "PRICE", "QTY", "NORD");
        ss << buffer;
    }
    ss << "================================================================\n";
    for (size_t l{0}; l < count and l < window; ++l)
    {
        auto price{market_data_message_->GetMbpAskPrice(count - l - 1)};
        auto qty{market_data_message_market_by_price_.ask_qty[count - l - 1]};
        {
            char buffer[256];
            sprintf(buffer, "%16.2f        [%8.2f]        (%8d)\n", price, qty, 1);
            ss << buffer;
        }
    }
    ss << "----------------------------------------------------------------\n";
    for (size_t l{0}; l < count and l < window; ++l)
    {
        auto price{market_data_message_->GetMbpBidPrice(l)};
        auto qty{market_data_message_market_by_price_.bid_qty[l]};
        {
            char buffer[256];
            sprintf(buffer, "%16.2f        [%8.2f]        (%8d)\n", price, qty, 1);
            ss << buffer;
        }
    }
    ss << "================================================================\n";

    return ss.str();
}

// check
bool MarketByPriceBook::IsValid() const
{
    return IsValidBid() && IsValidAsk() && GetPrice(ASK) > GetPrice(BID);
}

bool MarketByPriceBook::IsValidBid() const
{
    return GetPrice(BID) > INVALID_BID_PRICE;
}

bool MarketByPriceBook::IsValidAsk() const
{
    return GetPrice(ASK) < INVALID_ASK_PRICE;
}

bool MarketByPriceBook::IsFlip() const
{
    return false;
}

bool MarketByPriceBook::IsFlipUp() const
{
    return GetPrice(BID) >= prev_touch_price_[Ask];
}

bool MarketByPriceBook::IsFlipDown() const
{
    return GetPrice(ASK) <= prev_touch_price_[Bid];
}
}  // namespace alphaone
