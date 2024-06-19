#ifndef _TWSE_DATA_FILE_READER_H_
#define _TWSE_DATA_FILE_READER_H_

#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"

#include <sstream>
#include <sys/stat.h>

namespace alphaone
{

enum class TWSEDataReportType : int
{
    None,
    Add,
    AdjustVolume,
    AdjustPrice,
    Cancel,
    Trade,
};

std::string FromTWSEDataReportTypeToString(TWSEDataReportType type);

enum class TWSEDataReportTimeRestriction : char
{
    ROD = '0',
    IOC = '3',
    FOK = '4',
};

enum class TWSEDataReportTradeTypeCode : char
{
    Regular = '0',
    Block   = '1',
    Odd     = '2',
};

struct TWSEDataReport
{
    Timestamp                     Time;
    std::string_view              SecuritiesCode;
    std::string_view              OrderNumber;
    std::string_view              BrokerCode;
    std::string_view              TradeNumber;
    TWSEDataReportType            Type;
    BookSide                      BuySell;
    double                        Price;
    int                           Volume;
    TWSEDataReportTimeRestriction TimeRestriction;
    TWSEDataReportTradeTypeCode   TradeTypeCode;
    void *                        Raw;
};

struct FileDescription
{
    FileDescription()
        : Filename{""}
        , FD{0}
        , Addr{nullptr}
        , Offset{0}
        , FileOffset{0}
        , PaOffset{0}
        , Length{0}
        , LineLength{0}
    {
    }
    std::string Filename;
    int         FD;
    struct stat SB;
    char *      Addr;
    off_t       Offset;
    off_t       FileOffset;
    off_t       PaOffset;
    size_t      Length;
    size_t      LineLength;
};

template <typename Format>
bool ParserOrder(const Format *format, TWSEDataReport &report)
{
    std::string t(format->OrderTime, sizeof(format->OrderTime));
    t.insert(2, ":");
    t.insert(5, ":");
    t.insert(8, ".");

    Date d{Date::from_date_str(std::string(format->OrderDate, sizeof(format->OrderDate)).c_str())};

    report.Time           = Timestamp::from_date_time(d, t.c_str());
    report.SecuritiesCode = std::string_view(format->SecuritiesCode, 6);
    report.OrderNumber    = std::string_view(format->OrderNumber2, 5);
    report.BrokerCode     = std::string_view(format->OrderNumber1, 4);
    report.BuySell        = format->BuySell[0] == 'B';
    report.Price          = std::stod(std::string(format->OrderPrice, sizeof(format->OrderPrice)));
    report.Volume         = std::stoi(
        std::string(format->ChangedtheTradeVolume, sizeof(format->ChangedtheTradeVolume)));
    report.TimeRestriction = TWSEDataReportTimeRestriction::ROD;
    report.Raw             = const_cast<Format *>(format);

    switch (format->ChangedTradeCode[0])
    {
    case '1':
    case '4':
        report.Type = TWSEDataReportType::Add;
        break;
    case '2':  // volume
    case '5':  // volume
        report.Type = TWSEDataReportType::AdjustVolume;
        break;
    case '7':  // price
    case '8':  // price
        report.Type = TWSEDataReportType::AdjustPrice;
        break;
    case '3':
    case '6':
        report.Type = TWSEDataReportType::Cancel;
        break;
    default:
        return false;
    }

    switch (format->TradeTypeCode[0])
    {
    case '0':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Regular;
        break;
    case '1':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Block;
        break;
    case '2':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Odd;
        break;
    default:
        SPDLOG_CRITICAL("received unrecognized TradeTypeCode={}", format->TradeTypeCode[0]);
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Regular;
    }

    return true;
}
template <typename Format>
bool ParserOrder63(const Format *format, TWSEDataReport &report)
{
    auto res = ParserOrder<>(format, report);

    report.TimeRestriction = static_cast<TWSEDataReportTimeRestriction>(format->TimeRestriction[0]);

    return res;
}

template <typename Format>
bool ParserMatch(const Format *format, TWSEDataReport &report)
{
    std::string t(format->TradeTime, sizeof(format->TradeTime));
    t.insert(2, ":");
    t.insert(5, ":");
    t.insert(8, ".");

    Date d{Date::from_date_str(std::string(format->Date, sizeof(format->Date)).c_str())};

    report.Time           = Timestamp::from_date_time(d, t.c_str());
    report.SecuritiesCode = std::string_view(format->SecuritiesCode, 6);
    report.OrderNumber    = std::string_view(format->OrderNumber2, 5);
    report.BrokerCode     = std::string_view(format->OrderNumber1, 4);
    report.TradeNumber    = std::string_view(format->TradeNumber, 8);
    report.BuySell        = format->BuySell[0] == 'B' ? Side::Bid : Side::Ask;
    report.Price          = std::stod(std::string(format->TradePrice, sizeof(format->TradePrice)));
    report.Volume = std::stoi(std::string(format->TradeVolume, sizeof(format->TradeVolume)));
    report.Type   = TWSEDataReportType::Trade;
    report.TimeRestriction = TWSEDataReportTimeRestriction::ROD;
    report.Raw             = const_cast<Format *>(format);

    switch (format->TradeTypeCode[0])
    {
    case '0':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Regular;
        break;
    case '1':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Block;
        break;
    case '2':
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Odd;
        break;
    default:
        SPDLOG_CRITICAL("received unrecognized TradeTypeCode={}", format->TradeTypeCode[0]);
        report.TradeTypeCode = TWSEDataReportTradeTypeCode::Regular;
    }

    return true;
}

template <typename Format>
bool ParserMatch67(const Format *format, TWSEDataReport &report)
{
    auto res = ParserMatch<>(format, report);

    report.TimeRestriction = static_cast<TWSEDataReportTimeRestriction>(format->TimeRestriction[0]);

    return res;
}

class BaseDecoder
{
  public:
    BaseDecoder()          = default;
    virtual ~BaseDecoder() = default;


    virtual bool ReadOnce(TWSEDataReport &report) = 0;
    virtual void MoveNext()                       = 0;
};

template <typename Format, bool (*Parser)(const Format *data, TWSEDataReport &report)>
class Decoder : public BaseDecoder
{
  public:
    Decoder(FileDescription &desc)
        : BaseDecoder()
        , offset_{0}
        , format_size_{sizeof(Format)}
        , line_size_{format_size_ + 1}
        , desc_{desc}
    {
    }
    ~Decoder() = default;

    bool ReadOnce(TWSEDataReport &report) override
    {
        if (offset_ + format_size_ <= desc_.Length &&
            Parser((Format *)(&desc_.Addr[offset_]), report))
        {
            return true;
        }
        return false;
    }

    void MoveNext() override
    {
        offset_ += line_size_;
    }

  private:
    size_t          offset_;
    const size_t    format_size_;
    const size_t    line_size_;
    FileDescription desc_;
};


class TWSEDataFileReader
{
  public:
    TWSEDataFileReader(const std::string &order_filename, const std::string &match_filename);
    ~TWSEDataFileReader();

    bool ReadNext();

    const TWSEDataReport &Get() const;
    const Timestamp &     PeekTimestamp();

  private:
    enum class DataFileType : int
    {
        Order,
        Match,
    };

    const std::string order_filename_;
    const std::string match_filename_;

    FileDescription order_desc_;
    FileDescription match_desc_;

    TWSEDataReport next_order_report_;
    TWSEDataReport next_match_report_;

    BaseDecoder *order_decoder_;
    BaseDecoder *match_decoder_;

    bool has_next_order_;
    bool has_next_match_;

    void Setup(const std::string &filename, FileDescription &desc, DataFileType type);
};

}  // namespace alphaone

#endif
