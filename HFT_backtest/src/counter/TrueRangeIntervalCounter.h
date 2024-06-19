#ifndef _TRUERANGEINTERVALCOUNTER_H_
#define _TRUERANGEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/ObjectManager.h"

namespace alphaone
{
class TrueRangeIntervalCounter : public Counter
{
  public:
    TrueRangeIntervalCounter(const ObjectManager *object_manager,
                             MultiBookManager *multi_book_manager, Engine *engine,
                             const nlohmann::json &spec);
    TrueRangeIntervalCounter(const TrueRangeIntervalCounter &) = delete;
    TrueRangeIntervalCounter &operator=(const TrueRangeIntervalCounter &) = delete;
    TrueRangeIntervalCounter(TrueRangeIntervalCounter &&)                 = delete;
    TrueRangeIntervalCounter &operator=(TrueRangeIntervalCounter &&) = delete;

    ~TrueRangeIntervalCounter();

    void OnTrade(const Timestamp event_loop_time, const BookDataMessageTrade *o) final;
    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) final;

    std::string Name() const override;
    void        DumpDetail() const;

    void WarmUp() final;

  protected:
    void SetElements() override;

  private:
    nlohmann::json clock_time_counter_spec_;

    Counter *clock_time_counter_;
    size_t   last_clock_time_count_;

    const double tick_;
    const double tick_inverse_;

    BookPrice last_close_;
    BookPrice this_close_;
    double    last_volatility_;
    double    this_volatility_;
};
}  // namespace alphaone

#endif
