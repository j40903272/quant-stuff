#include "MarkoutTrackRecord.h"

namespace alphaone
{
MarkoutTrackRecord::MarkoutTrackRecord(const Book *book, const Counter *counter,
                                       const std::vector<size_t> &spacings, bool should_dump)
    : book_{book}
    , counter_{counter}
    , spacings_{spacings}
    , should_dump_{should_dump}
    , current_count_{0}
{
    tape_.reserve(1048576);
}

MarkoutTrackRecord::MarkoutTrackRecord(const Book *book, const Counter *counter, size_t length,
                                       bool should_dump)
    : MarkoutTrackRecord(book, counter, std::vector<size_t>({length}), should_dump)
{
}

MarkoutTrackRecord::~MarkoutTrackRecord()
{
    if (should_dump_)
    {
        char header[256];
        sprintf(header, "%32s %8s %16s %16s %16s \n", "type", "number", "markout_return",
                "markout_sharpe", "r2");
        for (const auto &length : spacings_)
        {
            std::stringstream ss;
            ss << "(" << counter_->ToString() << "|" << length << ")\n";
            ss << header;
            for (const auto &[type, _] : track_records_)
            {
                const auto number{GetCount(type)};
                const auto markout_return{GetMarkoutReturn(type, length)};
                const auto markout_sharpe{GetMarkoutSharpe(type, length)};
                const auto r2{GetMarkoutRSquared(type, length)};
                char       buffer[256];
                sprintf(buffer, "%32s %8ld %+16.8f %+16.8f %+16.8f \n", FromIdToType(type).c_str(),
                        number, markout_return, markout_sharpe, r2);
                ss << buffer;
            }
            std::cout << ss.str();
        }
    }
}

void MarkoutTrackRecord::Init(size_t key, const std::string &value)
{
    if (const std::map<size_t, std::string>::iterator it{from_id_to_type_map_.find(key)};
        it != from_id_to_type_map_.end())
    {
        abort();
    }
    from_id_to_type_map_.emplace(key, value);
}

std::string MarkoutTrackRecord::FromIdToType(size_t id) const
{
    const auto it{from_id_to_type_map_.find(id)};
    if (it == from_id_to_type_map_.end())
    {
        return std::to_string(id);
    }
    else
    {
        return it->second;
    }
}

void MarkoutTrackRecord::Update()
{
    if (BRANCH_LIKELY(book_->IsValid()))
    {
        const size_t diff{counter_->GetCount() - (current_count_ - 1)};
        if (diff > 0)
        {
            tape_.insert(tape_.end(), diff, book_->GetMidPrice());
            current_count_ += diff;
        }
    }
}

void MarkoutTrackRecord::Insert(const size_t type, const BookSide side, const size_t count,
                                const double price, const double prediction)
{
    track_records_[type].emplace_back(count, side, price, prediction);
}

void MarkoutTrackRecord::Insert(const size_t type, const BookSide side, const double price,
                                const double prediction)
{
    Insert(type, side, counter_->GetCount(), price, prediction);
}

void MarkoutTrackRecord::Insert(const size_t type, const BookSide side, const size_t count,
                                const double prediction)
{
    Insert(type, side, count, book_->GetMidPrice(), prediction);
}

void MarkoutTrackRecord::Insert(const size_t type, const BookSide side, const double prediction)
{
    Insert(type, side, counter_->GetCount(), book_->GetMidPrice(), prediction);
}

void MarkoutTrackRecord::Insert(const size_t type, const BookSide side)
{
    Insert(type, side, NaN);
}

size_t MarkoutTrackRecord::GetCount(const size_t type) const
{
    const std::map<size_t, std::vector<Record>>::const_iterator &it{track_records_.find(type)};
    if (it == track_records_.end())
    {
        return 0;
    }

    return it->second.size();
}

std::vector<double> MarkoutTrackRecord::GetMarkoutReturnSequence(const size_t type,
                                                                 const size_t length) const
{
    const std::map<size_t, std::vector<Record>>::const_iterator &it{track_records_.find(type)};
    if (it == track_records_.end())
    {
        return std::vector<double>();
    }
    const std::vector<Record> &record{it->second};

    std::vector<double> result;
    for (const auto item : record)
    {
        const size_t id{std::min(tape_.size() - 1, item.count_ + length)};
        const double rt{(item.side_ == Ask ? -1.0 : +1.0) * y_log(tape_[id] / item.price_)};
        result.emplace_back(rt);
    }

    return result;
}

std::vector<double> MarkoutTrackRecord::GetMarkoutReturnSequence(const size_t type) const
{
    if (spacings_.size() == 0)
    {
        SPDLOG_WARN("spacings.size()={}", spacings_.size());
        return std::vector<double>();
    }
    return GetMarkoutReturnSequence(type, spacings_[0]);
}

double MarkoutTrackRecord::GetMarkoutReturn(const size_t type, const size_t length) const
{
    const std::map<size_t, std::vector<Record>>::const_iterator &it{track_records_.find(type)};
    if (it == track_records_.end())
    {
        return 0.0;
    }
    const std::vector<Record> &record{it->second};

    double sum_return{0.0};
    for (const auto item : record)
    {
        const size_t id{std::min(tape_.size() - 1, item.count_ + length)};
        const double rt{(item.side_ == Ask ? -1.0 : +1.0) * y_log(tape_[id] / item.price_)};
        sum_return += rt;
    }
    sum_return = sum_return / record.size();

    const double markout_return{sum_return};
    return markout_return;
}

double MarkoutTrackRecord::GetMarkoutSharpe(const size_t type, const size_t length) const
{
    const std::map<size_t, std::vector<Record>>::const_iterator &it{track_records_.find(type)};
    if (it == track_records_.end())
    {
        return 0.0;
    }
    const std::vector<Record> &record{it->second};

    double sum_return{0.0};
    double sum_squared_return{0.0};
    for (const auto item : record)
    {
        const size_t id{std::min(tape_.size() - 1, item.count_ + length)};
        const double rt{(item.side_ == Ask ? -1.0 : +1.0) * y_log(tape_[id] / item.price_)};
        sum_return += rt;
        sum_squared_return += rt * rt;
    }
    sum_return         = sum_return / record.size();
    sum_squared_return = sum_squared_return / record.size();

    const double markout_sharpe{sum_return / std::sqrt(sum_squared_return)};
    return markout_sharpe;
}

double MarkoutTrackRecord::GetMarkoutRSquared(const size_t type, const size_t length) const
{
    const std::map<size_t, std::vector<Record>>::const_iterator &it{track_records_.find(type)};
    if (it == track_records_.end())
    {
        return 0.0;
    }
    const std::vector<Record> &record{it->second};

    double sum_return{0.0};
    // instead of comparing predicted values with the mean, we compare predicted values to 0.0
    /*
    for (const auto item : record)
    {
        const size_t id{std::min(tape_.size() - 1, item.count_ + length)};
        const double rt{y_log(tape_[id] / item.price_)};
        sum_return += rt;
    }
    sum_return = sum_return / record.size();
    */

    double sum_square{0.0};
    double sum_squared_residual{0.0};
    for (const auto item : record)
    {
        const size_t id{std::min(tape_.size() - 1, item.count_ + length)};
        const double rt{y_log(tape_[id] / item.price_)};
        const double pd{item.prediction_};
        sum_square += (rt - sum_return) * (rt - sum_return);
        sum_squared_residual += (rt - pd) * (rt - pd);
    }

    const double r2{1.0 - sum_squared_residual / sum_square};
    return r2;
}
}  // namespace alphaone
