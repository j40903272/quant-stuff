#ifndef _VOLATILITYINTERVALCOUNTER_H_
#define _VOLATILITYINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class VolatilityIntervalCounter : public Counter
{
  public:
    VolatilityIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                              const nlohmann::json &spec);
    VolatilityIntervalCounter(const VolatilityIntervalCounter &) = delete;
    VolatilityIntervalCounter &operator=(const VolatilityIntervalCounter &) = delete;
    VolatilityIntervalCounter(VolatilityIntervalCounter &&)                 = delete;
    VolatilityIntervalCounter &operator=(VolatilityIntervalCounter &&) = delete;

    ~VolatilityIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) final;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    enum Signal
    {
        MidPrice      = 0,
        WeightedPrice = 1
    };

    static inline double square(double x)
    {
        return x * x;
    }

    const Symbol *symbol_;
    const Signal  price_type_;
    const double  accuracy_;
    const double  given_tick_;
    const double  tick_;
    const double  tick_squared_inverse_;

    BookPrice last_price_;
    BookPrice this_price_;
    double    last_volatility_;
    double    this_volatility_;
};
}  // namespace alphaone

#endif
