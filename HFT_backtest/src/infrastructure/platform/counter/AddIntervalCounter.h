#ifndef _ADDINTERVALCOUNTER_H_
#define _ADDINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class AddIntervalCounter : public Counter
{
  public:
    AddIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    AddIntervalCounter(const AddIntervalCounter &) = delete;
    AddIntervalCounter &operator=(const AddIntervalCounter &) = delete;
    AddIntervalCounter(AddIntervalCounter &&)                 = delete;
    AddIntervalCounter &operator=(AddIntervalCounter &&) = delete;

    ~AddIntervalCounter();

    void OnPostBookAdd(const Timestamp event_loop_time, const BookDataMessageAdd *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
};
}  // namespace alphaone

#endif
