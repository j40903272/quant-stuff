#ifndef _DELETEINTERVALCOUNTER_H_
#define _DELETEINTERVALCOUNTER_H_

#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class DeleteIntervalCounter : public Counter
{
  public:
    DeleteIntervalCounter(const Book *book, MultiBookManager *multi_book_manager);
    DeleteIntervalCounter(const DeleteIntervalCounter &) = delete;
    DeleteIntervalCounter &operator=(const DeleteIntervalCounter &) = delete;
    DeleteIntervalCounter(DeleteIntervalCounter &&)                 = delete;
    DeleteIntervalCounter &operator=(DeleteIntervalCounter &&) = delete;

    ~DeleteIntervalCounter();

    void OnPostBookDelete(const Timestamp event_loop_time, const BookDataMessageDelete *o) override;

    std::string Name() const override;
    void        DumpDetail() const;

  private:
};
}  // namespace alphaone

#endif
