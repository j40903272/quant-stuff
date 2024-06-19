#ifndef DELIMITEDFILEREADER_H
#define DELIMITEDFILEREADER_H

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace alphaone
{
class DelimitedFileReader
{
  public:
    // will exit if file does not exist, or has no header when header argument = true
    DelimitedFileReader(const std::string &filename, const std::string &delimiters = ",",
                        const bool header = true);
    ~DelimitedFileReader();

    bool ReadNext();  // will return true as long as there's a new line to read

    // fetch
    const std::string &RetrieveString(const std::string &header_field) const;
    double             RetrieveDouble(const std::string &header_field) const;
    float              RetrieveFloat(const std::string &header_field) const;
    int                RetrieveInt(const std::string &header_field) const;
    long               RetrieveLong(const std::string &header_field) const;
    bool               RetrieveBool(const std::string &header_field) const;

    Timestamp
    GetTimestampDateWithTime(const Date &       date,
                             const std::string &header_field) const;  // date, hh:mm:ss.xxx[xxxxxx]
    Timestamp
    GetTimestampDateTime(const std::string &header_field) const;  // yyyy-mm-dd hh:mm:ss.xxx[xxxxxx]

    // check what headers exist
    bool HasHeader(const std::string &h) const;

    // get all the headers parsed in the constructor
    const std::vector<std::string> &GetHeaders() const;

    const std::string &GetHeaderString() const;

    const std::string &GetRow() const;

    const std::vector<std::string> &GetSplitEntries() const;

    inline std::string GetFilename()
    {
        return filename_;
    }

  private:
    const std::string filename_;
    const std::string delimiters_;  // delimiters we split by


    std::istream *is_;  // our input stream

    std::string row_;  // last row read

    // headers
    std::vector<std::string> header_;
    size_t                   header_size_;
    std::string              header_string_;

    // map from a column name to an index
    std::unordered_map<std::string, size_t>                 headers_and_positions_;
    std::unordered_map<std::string, size_t>::const_iterator headers_and_positions_end_;

    // parsed line
    std::vector<std::string> split_entries_;
    size_t                   split_entries_size_;

    // blank line (because we return string references)
    const std::string empty_string_;
};
}  // namespace alphaone

#endif
