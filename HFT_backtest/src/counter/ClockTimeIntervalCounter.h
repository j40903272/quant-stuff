#ifndef _CLOCKTIMEINTERVALCOUNTER_H_
#define _CLOCKTIMEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"
#include "infrastructure/platform/engine/Engine.h"

namespace alphaone
{
class ClockTimeIntervalCounter : public Counter
{
  public:
    ClockTimeIntervalCounter(const Book *book, MultiBookManager *multi_book_manager, Engine *engine,
                             const nlohmann::json &spec);
    ClockTimeIntervalCounter(const ClockTimeIntervalCounter &) = delete;
    ClockTimeIntervalCounter &operator=(const ClockTimeIntervalCounter &) = delete;
    ClockTimeIntervalCounter(ClockTimeIntervalCounter &&)                 = delete;
    ClockTimeIntervalCounter &operator=(ClockTimeIntervalCounter &&) = delete;

    ~ClockTimeIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    Duration  duration_;
    Timestamp time_start_;
    Timestamp time_end_;
};
}  // namespace alphaone

#endif
