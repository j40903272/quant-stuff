#ifndef _TIMEINTERVALCOUNTER_H_
#define _TIMEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class TimeIntervalCounter : public Counter
{
  public:
    TimeIntervalCounter(const Book *book, MultiBookManager *multi_book_manager,
                        const nlohmann::json &spec);
    TimeIntervalCounter(const TimeIntervalCounter &) = delete;
    TimeIntervalCounter &operator=(const TimeIntervalCounter &) = delete;
    TimeIntervalCounter(TimeIntervalCounter &&)                 = delete;
    TimeIntervalCounter &operator=(TimeIntervalCounter &&) = delete;

    ~TimeIntervalCounter();

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) override;

    Timestamp GetTimestamp() const;

    std::string Name() const override;
    void        DumpDetail() const;

  protected:
    void SetElements() override;

  private:
    const double threshold_in_second_;
};
}  // namespace alphaone

#endif
