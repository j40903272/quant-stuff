#include "CryptoMarketDataMessageFileReader.h"

#include "infrastructure/common/util/Logger.h"
#include "infrastructure/common/util/LoggerFormat.h"

#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <unistd.h>

namespace alphaone
{
CryptoMarketDataMessageFileReader::CryptoMarketDataMessageFileReader(const std::string &filename)
    : filename_{filename}
    , market_data_file_message_{nullptr}
    , file_{nullptr}
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

    file_.open(filename_);

    if (file_.is_open()) {
        SPDLOG_INFO("FILE OPEN");
    }

    if (!file_)
    {
        SPDLOG_ERROR("fail to open {}", filename_);
        abort();
    }
}

CryptoMarketDataMessageFileReader::~CryptoMarketDataMessageFileReader()
{
    if (file_.is_open()) {
        file_.close();
    }
}

bool CryptoMarketDataMessageFileReader::ReadNext()
{
    if (!file_.eof())
    {
        // SPDLOG_INFO("FILE NOT EOF");
        std::string line;
        if (std::getline(file_, line, '\n')) // Read one line from the file
        {
            // SPDLOG_INFO("FILE GETLINE");
            try
            {
                // SPDLOG_INFO("READNEXT line: {}", line);
                protobuf_message_ = nlohmann::json::parse(line); // Parse the line as JSON
                return true;
            }
            catch (const nlohmann::json::parse_error& e)
            {
                SPDLOG_ERROR("JSON parsing error: {}", e.what());
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    // SPDLOG_INFO("FILE IS EOF");
    return false;
}


DataSourceID CryptoMarketDataMessageFileReader::GetDataSourceID() const
{
    SPDLOG_INFO("am i here?5 {}", static_cast<alphaone::DataSourceID>(10));
    // if (market_data_file_message_ != nullptr)
    // {
        // return static_cast<alphaone::DataSourceID>(market_data_file_message_->Header.DataSourceID);
        return static_cast<alphaone::DataSourceID>(10);
    // }
    // return DataSourceID::UNKNOWN;
}

Timestamp CryptoMarketDataMessageFileReader::PeekTimestamp() const
{
    if (!file_.eof())
    {
        // SPDLOG_INFO("{}", protobuf_message_);

        if (protobuf_message_.contains("t") && protobuf_message_["t"].is_number_integer())
        {
            int64_t providerTime = protobuf_message_["t"].get<int64_t>();
            SPDLOG_INFO("peeeeeeeeeek {}", providerTime);
            return Timestamp::from_epoch_nsec(providerTime);
        }
        else
        {
            SPDLOG_ERROR("Timestamp 't' not found or invalid in JSON data");
        }
    }

    return Timestamp::invalid();
}

// const protobuf::MarketDataMessage &CryptoMarketDataMessageFileReader::GetMarketDataMessage() const
const nlohmann::json &CryptoMarketDataMessageFileReader::GetMarketDataMessage() const
{
    return protobuf_message_;
}

const std::string &CryptoMarketDataMessageFileReader::GetFilename()
{
    return filename_;
}
}  // namespace alphaone
