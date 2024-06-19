#ifndef _MARKOUTTRACKRECORD_H_
#define _MARKOUTTRACKRECORD_H_

#include "infrastructure/base/Book.h"
#include "infrastructure/common/math/Math.h"
#include "infrastructure/common/util/String.h"
#include "infrastructure/platform/counter/Counter.h"

namespace alphaone
{
class MarkoutTrackRecord
{
  public:
    MarkoutTrackRecord() = delete;
    MarkoutTrackRecord(const Book *book, const Counter *counter,
                       const std::vector<size_t> &spacings    = std::vector<size_t>{},
                       bool                       should_dump = false);
    MarkoutTrackRecord(const Book *book, const Counter *counter, size_t length,
                       bool should_dump = false);
    MarkoutTrackRecord(const MarkoutTrackRecord &) = delete;
    MarkoutTrackRecord &operator=(const MarkoutTrackRecord &) = delete;
    MarkoutTrackRecord(MarkoutTrackRecord &&)                 = delete;
    MarkoutTrackRecord &operator=(MarkoutTrackRecord &&) = delete;

    ~MarkoutTrackRecord();

    void Init(size_t key, const std::string &value);

    void Update();
    void Insert(const size_t type, const BookSide side, const size_t count, const double price,
                const double prediction);
    void Insert(const size_t type, const BookSide side, const double price,
                const double prediction);
    void Insert(const size_t type, const BookSide side, const size_t count,
                const double prediction);
    void Insert(const size_t type, const BookSide side, const double prediction);
    void Insert(const size_t type, const BookSide side);

    size_t              GetCount(const size_t type) const;
    std::vector<double> GetMarkoutReturnSequence(const size_t type, const size_t length) const;
    std::vector<double> GetMarkoutReturnSequence(const size_t type) const;
    double              GetMarkoutReturn(const size_t type, const size_t length) const;
    double              GetMarkoutSharpe(const size_t type, const size_t length) const;
    double              GetMarkoutRSquared(const size_t type, const size_t length) const;

  private:
    struct Record
    {
        Record(size_t count, BookSide side, double price, double prediction) noexcept
            : count_{count}, side_{side}, price_{price}, prediction_{prediction}
        {
        }

        size_t   count_;
        BookSide side_;
        double   price_;
        double   prediction_;
    };

    std::string FromIdToType(size_t id) const;

    const Book *              book_;
    const Counter *           counter_;
    const std::vector<size_t> spacings_;
    const bool                should_dump_;

    std::map<size_t, std::string>         from_id_to_type_map_;
    std::map<size_t, std::vector<Record>> track_records_;
    size_t                                current_count_;
    std::vector<double>                   tape_;
};
}  // namespace alphaone

#endif
