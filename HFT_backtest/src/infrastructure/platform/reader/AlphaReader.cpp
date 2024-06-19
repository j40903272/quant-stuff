#include "AlphaReader.h"

namespace alphaone
{

AlphaReader::AlphaReader(const std::string &alpha_path, size_t fit_length, const Date &date,
                         const std::string &name)
    : alpha_path_{alpha_path}
    , date_{date}
    , name_{name}
    , fit_length_{fit_length}
    , file_offset_{0}
    , pa_offset_{file_offset_ & ~(sysconf(_SC_PAGE_SIZE) - 1)}
{
    if (alpha_path_.has_filename())
    {
        if (!std::filesystem::exists(alpha_path))
        {
            SPDLOG_ERROR("alpha cache file = {} does not exist.", alpha_path);
            throw std::invalid_argument("file not exist");
        }
    }
    else
    {
        auto file_name = date_.to_string() + "." + name_ + ".alphalogger";
        alpha_path_    = alpha_path_ / file_name;
        if (!std::filesystem::exists(alpha_path))
        {
            SPDLOG_ERROR("alpha cache file = {} does not exist.", alpha_path);
            throw std::invalid_argument("file not exist");
        }
    }

    fd_ = open(alpha_path_.c_str(), O_RDONLY);

    if (fd_ < 0)
    {
        SPDLOG_ERROR("open file failed");
        throw std::runtime_error("open file failed");
    }

    if (fstat(fd_, &sb_) || !sb_.st_size)
    {
        SPDLOG_ERROR("file stat error or file size={}", sb_.st_size);
        // exit(EXIT_FAILURE);
    }

    length_      = sb_.st_size - file_offset_;
    predictions_ = reinterpret_cast<char *>(
        mmap(NULL, length_ + file_offset_ - pa_offset_, PROT_READ, MAP_PRIVATE, fd_, pa_offset_));
}

AlphaReader::~AlphaReader()
{
    munmap(predictions_, length_ + file_offset_ - pa_offset_);
    close(fd_);
}

double AlphaReader::GetPrediction(size_t packet_end_count, size_t fit_id)
{
    if (BRANCH_UNLIKELY(!packet_end_count))
    {
        SPDLOG_ERROR("packet end count should greater than 1");
        return 0.;
    }
    return reinterpret_cast<double *>(predictions_ + (packet_end_count - 1) * fit_length_ *
                                                         sizeof(double))[fit_id];
}

double *AlphaReader::GetPredictions(size_t packet_end_count)
{
    if (BRANCH_UNLIKELY(!packet_end_count))
    {
        SPDLOG_ERROR("packet end count should greater than 1");
        return nullptr;
    }
    return reinterpret_cast<double *>(predictions_ +
                                      (packet_end_count - 1) * fit_length_ * sizeof(double));
}

}  // namespace alphaone
