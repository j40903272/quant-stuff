#ifndef _WEIGHTEDTICKINTERVALCOUNTER_H_
#define _WEIGHTEDTICKINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class WeightedTickIntervalCounter : public Counter
{
  public:
    WeightedTickIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                                const nlohmann::json &spec);
    WeightedTickIntervalCounter(const WeightedTickIntervalCounter &) = delete;
    WeightedTickIntervalCounter &operator=(const WeightedTickIntervalCounter &) = delete;
    WeightedTickIntervalCounter(WeightedTickIntervalCounter &&)                 = delete;
    WeightedTickIntervalCounter &operator=(WeightedTickIntervalCounter &&) = delete;

    ~WeightedTickIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

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

    const Signal price_type_;
    const double accuracy_;
    const double tick_;
    const double tick_inverse_;

    BookPrice last_price_;
    BookPrice this_price_;
    double    last_path_;
    double    this_path_;
};
}  // namespace alphaone

#endif
