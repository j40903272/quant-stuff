#include "MarketDataMessageFileReader.h"

#include "infrastructure/common/util/Logger.h"
#include "infrastructure/common/util/LoggerFormat.h"

#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <unistd.h>

namespace alphaone
{
MarketDataMessageFileReader::MarketDataMessageFileReader(const std::string &filename)
    : filename_{filename}
    , market_data_file_message_{nullptr}
    , file_{nullptr}
    , gzfile_{nullptr}
    , is_eof_{false}
    , bounds_{44, 186}
    , memory_{nullptr}
    , offset_{0}
    , file_offset_{0}
    , page_offset_{0}
    , length_{0}
{
    if (!std::filesystem::exists(filename_))
    {
        SPDLOG_ERROR("filename={} does not exist.", filename_);
        abort();
    }

    if (filename.length() > 3 && filename.substr(filename.length() - 3) == ".gz")
    {
        gzfile_ = gzopen(filename_.c_str(), "rb");
    }
    else
    {
        file_ = fopen(filename_.c_str(), "rb");
    }

    if (!gzfile_ && !file_)
    {
        SPDLOG_ERROR("fail to open {}", filename_);
        abort();
    }

    if (file_)
    {
        int         fd{fileno(file_)};
        struct stat sb;
        if (fstat(fd, &sb) || !sb.st_size)
        {
            SPDLOG_ERROR("error file stat with file size={}", sb.st_size);
            abort();
        }

        page_offset_ = file_offset_ & ~(sysconf(_SC_PAGE_SIZE) - 1);
        length_      = sb.st_size;
        memory_      = reinterpret_cast<char *>(
            mmap(NULL, length_ - page_offset_, PROT_READ, MAP_PRIVATE, fd, page_offset_));
    }
    else
    {
        char data[65535];
        while (!gzeof(gzfile_))
        {
            length_ += gzread(gzfile_, &data[0], 65535);
        }
        gzrewind(gzfile_);
        SPDLOG_DEBUG("length={}", length_);

        off_t offset{0};
        memory_ = new char[length_];
        while (!gzeof(gzfile_))
        {
            offset += gzread(gzfile_, memory_ + offset, 65535);
        }
    }
}

MarketDataMessageFileReader::~MarketDataMessageFileReader()
{
    if (gzfile_)
    {
        delete[] memory_;
        gzclose(gzfile_);
    }

    if (file_)
    {
        munmap(memory_, length_ - page_offset_);
        fclose(file_);
    }
}

bool MarketDataMessageFileReader::ReadNext()
{
    if (!is_eof_)
    {
        while (true)
        {
            market_data_file_message_ = (MarketDataFileStruct *)&memory_[offset_];

            if (market_data_file_message_->Header.DataLength < bounds_[0] ||
                market_data_file_message_->Header.DataLength > bounds_[1])
            {
                ++offset_;
                if (offset_ + static_cast<off_t>(sizeof(MarketDataFileStruct)) >= length_)
                {
                    is_eof_ = true;
                }

                SPDLOG_CRITICAL("length={} out of bound from {} to {}",
                                market_data_file_message_->Header.DataLength, bounds_[0],
                                bounds_[1]);

                return false;
            }
            else
            {
                break;
            }
        }

        stream_.rdbuf()->pubsetbuf(&market_data_file_message_->Data[0],
                                   market_data_file_message_->Header.DataLength);
        // protobuf_message_.ParseFromIstream(&stream_);
        stream_.str("");
        stream_.clear();

        offset_ += sizeof(MarketDataFileHeader) + market_data_file_message_->Header.DataLength;
        if (offset_ >= length_)
        {
            is_eof_ = true;
        }

        return true;
    }
    else
    {
        return false;
    }
}

DataSourceID MarketDataMessageFileReader::GetDataSourceID() const
{
    SPDLOG_INFO("am i here?8");
    if (market_data_file_message_ != nullptr)
    {
        return static_cast<alphaone::DataSourceID>(market_data_file_message_->Header.DataSourceID);
    }
    return DataSourceID::UNKNOWN;
}

Timestamp MarketDataMessageFileReader::PeekTimestamp() const
{
    SPDLOG_INFO("PeekTimestamp??123123123123");
    if (!is_eof_)
    {
        return Timestamp::from_epoch_nsec(
            ((MarketDataFileStruct *)&memory_[offset_])->Header.ProviderTime);
    }

    return Timestamp::invalid();
}

// const protobuf::MarketDataMessage &MarketDataMessageFileReader::GetMarketDataMessage() const
const nlohmann::json &MarketDataMessageFileReader::GetMarketDataMessage() const
{
    return protobuf_message_;
}

const std::string &MarketDataMessageFileReader::GetFilename()
{
    return filename_;
}
}  // namespace alphaone
