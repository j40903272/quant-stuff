#ifndef _SYMBOLINFO_H_
#define _SYMBOLINFO_H_

#include "infrastructure/base/Ohlcv.h"
#include "infrastructure/common/symbol/Symbol.h"

namespace alphaone
{
class ReferenceDataReader;

// this structure should align with ohlcv headers
class SymbolInfo
{
  public:
    SymbolInfo(int retrieving_days = 0)
        : open_{std::nan("")}
        , high_{std::nan("")}
        , low_{std::nan("")}
        , close_{std::nan("")}
        , volume_{std::nan("")}
        , sum_touch_size_{std::nan("")}
        , square_sum_touch_size_{std::nan("")}
        , sum_trade_size_{std::nan("")}
        , square_sum_trade_size_{std::nan("")}
        , sum_spread_{std::nan("")}
        , square_sum_spread_{std::nan("")}
        , sum_daily_volume_{std::nan("")}
        , read_days_{0}
        , book_count_{0}
        , trade_count_{0}
    {
        ohlcvs_.reserve(retrieving_days);
    }

    SymbolInfo &operator=(const Ohlcv<BookPrice, BookQty> &o)
    {
        open_                  = o.open_;
        high_                  = o.high_;
        low_                   = o.low_;
        close_                 = o.close_;
        volume_                = o.volume_;
        sum_daily_volume_      = o.volume_;
        sum_touch_size_        = o.sum_touch_size_;
        square_sum_touch_size_ = o.square_sum_touch_size_;
        sum_trade_size_        = o.sum_trade_size_;
        square_sum_trade_size_ = o.square_sum_trade_size_;
        sum_spread_            = o.sum_spread_;
        square_sum_spread_     = o.square_sum_spread_;
        read_days_             = 1;
        book_count_            = o.book_count_;
        trade_count_           = o.trade_count_;
        return *this;
    }

    inline double GetLastOpen() const
    {
        return open_;
    }

    inline double GetLastHigh() const
    {
        return high_;
    }

    inline double GetLastLow() const
    {
        return low_;
    }

    inline double GetLastClose() const
    {
        return close_;
    }

    inline double GetLastVolume() const
    {
        return volume_;
    }

    inline double GetAvgTouchSize() const
    {
        return book_count_ > 0 ? sum_touch_size_ / book_count_ : std::nan("");
    }

    inline double GetVarTouchSize() const
    {
        return (book_count_ > 1)
                   ? (square_sum_touch_size_ - book_count_ * square(GetAvgTouchSize())) /
                         (book_count_ - 1)
                   : std::nan("");
    }

    inline double GetAvgTradeSize() const
    {
        return trade_count_ > 0 ? sum_trade_size_ / trade_count_ : std::nan("");
    }

    inline double GetVarTradeSize() const
    {
        return (trade_count_ > 1)
                   ? (square_sum_trade_size_ - trade_count_ * square(GetAvgTradeSize())) /
                         (trade_count_ - 1)
                   : std::nan("");
    }

    inline double GetAvgDailyVolume() const
    {
        return read_days_ > 0 ? sum_daily_volume_ / read_days_ : std::nan("");
    }

    inline double GetAvgSpread() const
    {
        return book_count_ > 0 ? sum_spread_ / book_count_ : std::nan("");
    }

    inline double GetVarSpread() const
    {
        return (book_count_ > 1)
                   ? (square_sum_spread_ - book_count_ * square(GetAvgSpread())) / (book_count_ - 1)
                   : std::nan("");
    }

    inline uint32_t GetReadDays() const
    {
        return read_days_;
    }

    inline uint32_t GetBookCount() const
    {
        return book_count_;
    }

    inline uint32_t GetTradeCount() const
    {
        return trade_count_;
    }

    // store in reversee date order
    inline const std::vector<Ohlcv<BookPrice, BookQty>> &GetOhlcvs() const
    {
        return ohlcvs_;
    }

    // if not find return nullptr
    inline const Ohlcv<BookPrice, BookQty> *GetOhlcv(int32_t date)
    {
        return date_to_ohlcv_[date];
    }

  protected:
    friend class ReferenceDataReader;

  private:
    std::function<double(double)> square = [](auto x) { return x * x; };

    double   open_;
    double   high_;
    double   low_;
    double   close_;
    double   volume_;
    double   sum_touch_size_;
    double   square_sum_touch_size_;
    double   sum_trade_size_;
    double   square_sum_trade_size_;
    double   sum_spread_;
    double   square_sum_spread_;
    double   sum_daily_volume_;
    uint32_t read_days_;
    uint32_t book_count_;
    uint32_t trade_count_;

    std::vector<Ohlcv<BookPrice, BookQty>>                   ohlcvs_;
    std::unordered_map<int32_t, Ohlcv<BookPrice, BookQty> *> date_to_ohlcv_;
};

struct ShareInfo
{
  public:
    ShareInfo(const Symbol *symbol, const std::string &pid, const std::string &create_date,
              const std::string &active_date, const long public_shares,
              const BookPrice &close_price, const BookPrice &ref_opening_price,
              const long market_cap, const double ref_weight, const int sector_class,
              const int next_sector_class, const long next_public_shares,
              const long next_market_cap, const double next_ref_weight, const std::string &memo)
        : symbol_{symbol}
        , pid_{pid}
        , create_date_{create_date}
        , active_date_{active_date}
        , public_shares_{public_shares}
        , close_price_{close_price}
        , ref_opening_price_{ref_opening_price}
        , market_cap_{market_cap}
        , ref_weight_{ref_weight}
        , sector_class_{sector_class}
        , next_sector_class_{next_sector_class}
        , next_public_shares_{next_public_shares}
        , next_market_cap_{next_market_cap}
        , next_ref_weight_{next_ref_weight}
        , memo_{memo}
    {
    }

    const Symbol *GetSymbol() const
    {
        return symbol_;
    }
    std::string GetPid() const
    {
        return pid_;
    }
    std::string GetCreateDate() const
    {
        return create_date_;
    }
    std::string GetActiveDate() const
    {
        return active_date_;
    }
    long GetPublicShares() const
    {
        return public_shares_;
    }
    BookPrice GetClosePrice() const
    {
        return close_price_;
    }
    BookPrice GetRefOpeningPrice() const
    {
        return ref_opening_price_;
    }
    long GetMarketCap() const
    {
        return market_cap_;
    }
    double GetRefWeight() const
    {
        return ref_weight_;
    }
    int GetSectorClass() const
    {
        return sector_class_;
    }
    int GetNextSectorClass() const
    {
        return next_sector_class_;
    }
    long GetNextPublicShares() const
    {
        return next_public_shares_;
    }
    long GetNextMarketCap() const
    {
        return next_market_cap_;
    }
    std::string GetMemo() const
    {
        return memo_;
    }

  private:
    const Symbol *    symbol_;
    const std::string pid_;
    const std::string create_date_;
    const std::string active_date_;
    const long        public_shares_;
    const BookPrice   close_price_;
    const BookPrice   ref_opening_price_;
    const long        market_cap_;
    const double      ref_weight_;
    const int         sector_class_;
    const int         next_sector_class_;
    const long        next_public_shares_;
    const long        next_market_cap_;
    const double      next_ref_weight_;
    const std::string memo_;
};

}  // namespace alphaone


#endif
