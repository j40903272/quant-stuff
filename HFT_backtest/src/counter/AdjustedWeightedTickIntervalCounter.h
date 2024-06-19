#ifndef _ADJUSTEDWEIGHTEDTICKINTERVALCOUNTER_H_
#define _ADJUSTEDWEIGHTEDTICKINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class AdjustedWeightedTickIntervalCounter : public Counter
{
  public:
    AdjustedWeightedTickIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                                        const nlohmann::json &spec);
    AdjustedWeightedTickIntervalCounter(const AdjustedWeightedTickIntervalCounter &) = delete;
    AdjustedWeightedTickIntervalCounter &
    operator=(const AdjustedWeightedTickIntervalCounter &)                      = delete;
    AdjustedWeightedTickIntervalCounter(AdjustedWeightedTickIntervalCounter &&) = delete;
    AdjustedWeightedTickIntervalCounter &operator=(AdjustedWeightedTickIntervalCounter &&) = delete;

    ~AdjustedWeightedTickIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) final;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    const double multiplier_;

    BookPrice last_adjusted_weighted_tick_price_;
};
}  // namespace alphaone

#endif
