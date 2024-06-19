#ifndef _OHLCV_H_
#define _OHLCV_H_

#include <math.h>
#include <stdint.h>

namespace alphaone
{

template <typename P, typename Q>
struct Ohlcv
{
    Ohlcv()
        : open_{std::nan("")}
        , high_{std::nan("")}
        , low_{std::nan("")}
        , close_{std::nan("")}
        , volume_{0}
        , sum_touch_size_{0}
        , square_sum_touch_size_{0}
        , sum_spread_{0}
        , square_sum_spread_{0}
        , sum_touch_nord_{0}
        , sum_trade_size_{0}
        , square_sum_trade_size_{0}
        , sum_trade_nord_{0}
        , book_count_{0}
        , trade_count_{0}
    {
    }

    Ohlcv(const P open, const P high, const P low, const P close, const Q volume,
          const Q sum_touch_size, const Q square_sum_touch_size, const P sum_spread,
          const P square_sum_spread, const Q sum_trade_size, const Q square_sum_trade_size,
          const uint32_t book_count, const uint32_t trade_count)
        : open_{open}
        , high_{high}
        , low_{low}
        , close_{close}
        , volume_{volume}
        , sum_touch_size_{sum_touch_size}
        , square_sum_touch_size_{square_sum_touch_size}
        , sum_spread_{sum_spread}
        , square_sum_spread_{square_sum_spread}
        , sum_touch_nord_{0}
        , sum_trade_size_{sum_trade_size}
        , square_sum_trade_size_{square_sum_trade_size}
        , sum_trade_nord_{0}
        , book_count_{book_count}
        , trade_count_{trade_count}
    {
    }

    P        open_;
    P        high_;
    P        low_;
    P        close_;
    Q        volume_;
    Q        sum_touch_size_;
    Q        square_sum_touch_size_;
    P        sum_spread_;
    P        square_sum_spread_;
    uint32_t sum_touch_nord_;
    Q        sum_trade_size_;
    Q        square_sum_trade_size_;
    uint32_t sum_trade_nord_;
    uint32_t book_count_;
    uint32_t trade_count_;
};


}  // namespace alphaone


#endif
