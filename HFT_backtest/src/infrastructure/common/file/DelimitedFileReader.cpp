#include "DelimitedFileReader.h"

#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/gzstream.h"

#include <boost/algorithm/string.hpp>  // split
#include <iostream>
#include <set>

namespace alphaone
{
DelimitedFileReader::DelimitedFileReader(const std::string &filename, const std::string &delims,
                                         const bool use_header)
    : filename_{filename}
    , delimiters_{delims}
    , is_{nullptr}
    , header_size_{0}
    , header_string_{""}
    , headers_and_positions_end_{headers_and_positions_.end()}
    , split_entries_size_{0}
    , empty_string_{""}
{
    if (filename.length() > 3 && filename.substr(filename.length() - 3) == ".gz")
    {
        is_ = new igzstream(filename.c_str(), std::ios::in);
    }
    else
    {
        is_ = new std::ifstream(filename);
    }

    if (is_ == nullptr)
    {
        SPDLOG_ERROR("fail to read filename={}", filename);
        abort();
    }

    if (is_->fail())
    {
        SPDLOG_ERROR("fail to read filename={}", filename);
        abort();
    }

    // read the header
    if (use_header)
    {
        std::getline(*is_, row_);
        header_string_ = row_;

        // parse header
        boost::split(header_, row_, boost::is_any_of(delims));
        header_size_ = header_.size();

        // error if duplicate header names exist, we cannot handle that
        // also create an unordered map to map from header name to ordinal position
        std::set<std::string> unique_header_names;
        for (size_t i{0}; i < header_size_; ++i)
        {
            if (unique_header_names.find(header_[i]) != unique_header_names.end())
            {
                SPDLOG_ERROR("cannot handle duplicated header name={}", header_[i]);
                abort();
            }
            unique_header_names.insert(header_[i]);
            headers_and_positions_[header_[i]] = i;
        }
        headers_and_positions_end_ = headers_and_positions_.end();
    }

    if (use_header && header_.empty())
    {
        SPDLOG_ERROR("file={} has empty header", filename);
        abort();
    }
}

DelimitedFileReader::~DelimitedFileReader()
{
    delete is_;
}

bool DelimitedFileReader::ReadNext()
{
    bool good{getline(*is_, row_) && (header_size_ == 0 || row_.length() > 0)};
    while (good && (row_.length() > 1 && row_.substr(0, 1) == "#"))  // keep going for comments
    {
        good = getline(*is_, row_) && (header_size_ == 0 || row_.length() > 0);
    }

    if (good)
    {
        split_entries_.clear();
        split_entries_.reserve(header_size_);
        boost::split(split_entries_, row_, boost::is_any_of(delimiters_));
        split_entries_size_ = split_entries_.size();
        if (header_size_ > 0 && header_size_ != split_entries_size_)
        {
            SPDLOG_WARN("cannot process line=[{}]... with split_entries_size={} but header_size={}",
                        row_.substr(0, 30), split_entries_size_, header_size_);
            return false;
        }
    }
    return good;
}

const std::string &DelimitedFileReader::RetrieveString(const std::string &header_field) const
{
    auto it{headers_and_positions_.find(header_field)};
    if (it == headers_and_positions_end_)
    {
        return empty_string_;
    }

    const size_t pos{it->second};
    if (pos > split_entries_size_)
    {
        SPDLOG_ERROR("cannot find field={}", header_field);
        abort();
    }
    return split_entries_[pos];
}

double DelimitedFileReader::RetrieveDouble(const std::string &header_field) const
{
    return std::stod(RetrieveString(header_field));
}

float DelimitedFileReader::RetrieveFloat(const std::string &header_field) const
{
    return std::stof(RetrieveString(header_field));
}

int DelimitedFileReader::RetrieveInt(const std::string &header_field) const
{
    return std::atoi(RetrieveString(header_field).c_str());
}

long DelimitedFileReader::RetrieveLong(const std::string &header_field) const
{
    return std::atol(RetrieveString(header_field).c_str());
}

bool DelimitedFileReader::RetrieveBool(const std::string &header_field) const
{
    if (header_field.size() == 0)
    {
        return false;
    }

    char ch{RetrieveString(header_field)[0]};
    return !(ch == '0' || ch == 'f' || ch == 'F' || ch == 'n' || ch == 'N');
}

Timestamp DelimitedFileReader::GetTimestampDateWithTime(const Date &       date,
                                                        const std::string &header_field) const
{
    return Timestamp::from_date_time(date, RetrieveString(header_field).c_str());
}

Timestamp DelimitedFileReader::GetTimestampDateTime(const std::string &header_field) const
{
    return Timestamp::from_date_time(RetrieveString(header_field).c_str());
}

const std::string &DelimitedFileReader::GetRow() const
{
    return row_;
}

const std::vector<std::string> &DelimitedFileReader::GetSplitEntries() const
{
    return split_entries_;
}

const std::vector<std::string> &DelimitedFileReader::GetHeaders() const
{
    return header_;
}

bool DelimitedFileReader::HasHeader(const std::string &header_field) const
{
    return headers_and_positions_.find(header_field) != headers_and_positions_end_;
}

const std::string &DelimitedFileReader::GetHeaderString() const
{
    return header_string_;
}
}  // namespace alphaone
