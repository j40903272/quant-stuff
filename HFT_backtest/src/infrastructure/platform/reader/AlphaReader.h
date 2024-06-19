#ifndef _ALPHAREADER_H_
#define _ALPHAREADER_H_

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/util/Branch.h"

#include <fcntl.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace alphaone
{

class AlphaReader
{
  public:
    AlphaReader(const std::string &alpha_path, size_t fit_length, const Date &date,
                const std::string &);
    ~AlphaReader();

    double  GetPrediction(size_t packet_end_count, size_t fit_id);
    double *GetPredictions(size_t packet_end_count);

  private:
    std::filesystem::path alpha_path_;

    const Date        date_;
    const std::string name_;
    const size_t      fit_length_;
    const off_t       file_offset_;
    const off_t       pa_offset_;
    int               fd_;
    struct stat       sb_;
    off_t             offset_;

    size_t length_;
    char * predictions_;
};


}  // namespace alphaone


#endif