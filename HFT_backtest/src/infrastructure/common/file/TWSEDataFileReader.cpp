#include "TWSEDataFileReader.h"

#include "infrastructure/common/message/TWSEDataFileFormat.h"

#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <unistd.h>

namespace alphaone
{
std::string FromTWSEDataReportTypeToString(TWSEDataReportType type)
{
    if (type == TWSEDataReportType::Add)
    {
        return "TWSEDataReportType::Add";
    }
    if (type == TWSEDataReportType::AdjustVolume)
    {
        return "TWSEDataReportType::AdjustVolume";
    }
    if (type == TWSEDataReportType::AdjustPrice)
    {
        return "TWSEDataReportType::AdjustPrice";
    }
    if (type == TWSEDataReportType::Cancel)
    {
        return "TWSEDataReportType::Cancel";
    }
    if (type == TWSEDataReportType::Trade)
    {
        return "TWSEDataReportType::Trade";
    }
    return "TWSEDataReportType::None";
}

TWSEDataFileReader::TWSEDataFileReader(const std::string &order_filename,
                                       const std::string &match_filename)
    : order_filename_{order_filename}
    , match_filename_{match_filename}
    , order_decoder_{nullptr}
    , match_decoder_{nullptr}
    , has_next_order_{false}
    , has_next_match_{false}
{
    if (!std::filesystem::exists(order_filename_))
    {
        SPDLOG_ERROR("filename={} does not exist.", order_filename_);
        exit(EXIT_FAILURE);
    }

    Setup(order_filename_, order_desc_, DataFileType::Order);

    if (!std::filesystem::exists(match_filename))
    {
        SPDLOG_ERROR("filename={} does not exist.", match_filename);
        exit(EXIT_FAILURE);
    }

    Setup(match_filename, match_desc_, DataFileType::Match);

    next_match_report_.TradeTypeCode   = TWSEDataReportTradeTypeCode::Regular;
    next_match_report_.TimeRestriction = TWSEDataReportTimeRestriction::ROD;
}

TWSEDataFileReader::~TWSEDataFileReader()
{
    munmap(order_desc_.Addr, order_desc_.Length + order_desc_.FileOffset - order_desc_.PaOffset);
    close(order_desc_.FD);

    munmap(match_desc_.Addr, match_desc_.Length + match_desc_.FileOffset - match_desc_.PaOffset);
    close(match_desc_.FD);

    if (match_decoder_ != nullptr)
    {
        delete match_decoder_;
        match_decoder_ = nullptr;
    }

    if (order_decoder_ != nullptr)
    {
        delete order_decoder_;
        order_decoder_ = nullptr;
    }
}

bool TWSEDataFileReader::ReadNext()
{
    has_next_order_ = order_decoder_->ReadOnce(next_order_report_);
    has_next_match_ = match_decoder_->ReadOnce(next_match_report_);
    return has_next_order_ || has_next_match_;
}

const TWSEDataReport &TWSEDataFileReader::Get() const
{
    if (has_next_order_ && has_next_match_)
    {
        if (next_order_report_.Time <= next_match_report_.Time)
        {
            order_decoder_->MoveNext();
            return next_order_report_;
        }
    }
    match_decoder_->MoveNext();
    return next_match_report_;
}

const Timestamp &TWSEDataFileReader::PeekTimestamp()
{
    SPDLOG_INFO("PeekTimestamp??WTF123");
    if (has_next_order_ && has_next_match_)
    {
        return next_order_report_.Time <= next_match_report_.Time ? next_order_report_.Time
                                                                  : next_match_report_.Time;
    }
    return next_match_report_.Time;
}

void TWSEDataFileReader::Setup(const std::string &filename, FileDescription &desc,
                               DataFileType type)
{
    int &        fd          = desc.FD;
    struct stat &sb          = desc.SB;
    off_t &      file_offset = desc.FileOffset;
    off_t &      pa_offset   = desc.PaOffset;
    size_t &     length      = desc.Length;
    char *&      addr        = desc.Addr;

    fd = open(filename.c_str(), O_RDONLY);

    if (fstat(fd, &sb) || !sb.st_size)
    {
        SPDLOG_ERROR("error file stat with file size={}", sb.st_size);
        exit(EXIT_FAILURE);
    }

    pa_offset = file_offset & ~(sysconf(_SC_PAGE_SIZE) - 1);
    length    = sb.st_size - file_offset;

    addr = reinterpret_cast<char *>(
        mmap(NULL, length + file_offset - pa_offset, PROT_READ, MAP_PRIVATE, fd, pa_offset));

    if (type == DataFileType::Order)
    {
        if (addr[sizeof(OrderReport63)] == '\n')
        {
            order_decoder_ = new Decoder<OrderReport63, ParserOrder63<OrderReport63>>(desc);
        }
        else
        {
            order_decoder_ = new Decoder<OrderReport59, ParserOrder<OrderReport59>>(desc);
        }
    }
    else if (type == DataFileType::Match)
    {
        if (addr[sizeof(MatchReport67)] == '\n')
        {
            match_decoder_ = new Decoder<MatchReport67, ParserMatch67<MatchReport67>>(desc);
        }
        else
        {
            match_decoder_ = new Decoder<MatchReport63, ParserMatch<MatchReport63>>(desc);
        }
    }
}

}  // namespace alphaone
