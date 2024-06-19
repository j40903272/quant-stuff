#include "ReferenceDataReader.h"

namespace alphaone
{

ReferenceDataReader::ReferenceDataReader(const SymbolManager *symbol_manager, nlohmann::json node)
    : symbol_manager_{symbol_manager}
    , retrieving_days_{node.value("retrieving_days", 15)}
    , least_trade_count_{node.value("least_trade_count", 60)}
    , file_name_{"reference_data.csv"}
    , first_date_{node.value("is_from_today", false) ? symbol_manager_->GetDate()
                                                     : symbol_manager_->GetDate() - 1}
    , root_path_{symbol_manager_->GetJsonPath().parent_path().parent_path()}
{
    Read();
}

ReferenceDataReader::~ReferenceDataReader()
{
}

void ReferenceDataReader::Read()
{
    const Date last_date = Date::from_yyyymmdd(20200526);  // start from yesterday, until 20200526
    Date       date      = first_date_;
    int        read_days = 0;
    while (read_days < retrieving_days_)
    {
        if (date < last_date)
        {
            SPDLOG_WARN("read exceeds last date = {}, stop reading reference data",
                        last_date.to_string());
            break;
        }

        std::filesystem::path ref_path{root_path_ / date.to_string_with_dash() / file_name_};
        try
        {
            if (!std::filesystem::exists(ref_path))
            {
                SPDLOG_DEBUG("skipping {} reference data due to no file is found",
                             date.to_string());
                --date;
                continue;
            }
        }
        catch (const std::exception &e)
        {
            SPDLOG_ERROR("catch exception {} thus skipping {}", e.what(), date.to_string());
            --date;
            continue;
        }

        SPDLOG_INFO("reading {} reference data", date.to_string());
        DelimitedFileReader reader  = DelimitedFileReader(ref_path.string());
        const auto          headers = reader.GetHeaders();
        const auto          columns = headers.size();

        // HACK: temporarily skip previous version before update
        if (columns != 14UL)
        {
            throw std::invalid_argument(fmt::format("invalid header size = {}", headers.size()));
        }
        // clang-format off
        // "symbol", "open", "high", "low", "close", "volume","sum_touch_size", "book_count", "sum_trade_size", "trade_count", "square_sum_touch_size", "square_sum_trade_size", "sum_spread", "square_sum_spread"
        //       0 ,     1 ,     2 ,    3 ,      4 ,       5 ,              6 ,           7 ,               8 ,            9 ,                     10 ,                     11 ,          12 ,                 13
        // clang-format on

        while (reader.ReadNext())
        {
            const Symbol *symbol =
                symbol_manager_->GetSymbolByString(reader.RetrieveString(headers[0]));

            if (reader.RetrieveInt("trade_count") < least_trade_count_)
                continue;  // skip trade count too small

            Ohlcv<BookPrice, BookQty> ohlcv{
                reader.RetrieveDouble("open"),
                reader.RetrieveDouble("high"),
                reader.RetrieveDouble("low"),
                reader.RetrieveDouble("close"),
                reader.RetrieveDouble("volume"),
                reader.RetrieveDouble("sum_touch_size"),
                reader.RetrieveDouble("square_sum_touch_size"),
                reader.RetrieveDouble("sum_spread"),
                reader.RetrieveDouble("square_sum_spread"),
                reader.RetrieveDouble("sum_trade_size"),
                reader.RetrieveDouble("square_sum_trade_size"),
                static_cast<uint32_t>(reader.RetrieveInt("book_count")),
                static_cast<uint32_t>(reader.RetrieveInt("trade_count"))};

            if (auto [it, is_success] =
                    symbol_to_info_map_.insert({symbol, SymbolInfo(retrieving_days_)});
                is_success)
            {
                it->second = ohlcv;
                it->second.ohlcvs_.emplace_back(ohlcv);
                it->second.date_to_ohlcv_[date.to_yyyymmdd()] = &it->second.ohlcvs_.back();
            }
            else
            {
                auto &symbol_info = it->second;
                if (!std::isnan(symbol_info.sum_daily_volume_))
                {
                    symbol_info.sum_daily_volume_ += ohlcv.volume_;
                    symbol_info.read_days_ += 1;
                    symbol_info.ohlcvs_.emplace_back(ohlcv);
                    symbol_info.date_to_ohlcv_[date.to_yyyymmdd()] = &symbol_info.ohlcvs_.back();
                }

                if (!std::isnan(symbol_info.sum_touch_size_))
                {
                    symbol_info.sum_touch_size_ += ohlcv.sum_touch_size_;
                    symbol_info.book_count_ += ohlcv.book_count_;
                    symbol_info.square_sum_touch_size_ += ohlcv.square_sum_touch_size_;
                }

                if (!std::isnan(symbol_info.sum_trade_size_))
                {
                    symbol_info.sum_trade_size_ += ohlcv.sum_trade_size_;
                    symbol_info.trade_count_ += ohlcv.trade_count_;
                    symbol_info.square_sum_trade_size_ += ohlcv.square_sum_trade_size_;
                }

                if (!std::isnan(symbol_info.sum_spread_))
                {
                    symbol_info.sum_spread_ += ohlcv.sum_spread_;
                    symbol_info.square_sum_spread_ += ohlcv.square_sum_spread_;
                }
            }
        }

        ++read_days;
        --date;
    }
}

const SymbolInfo &ReferenceDataReader::GetSymbolInfoFromSymbol(const Symbol *symbol) const
{
    if (auto it = symbol_to_info_map_.find(symbol); it != symbol_to_info_map_.end())
    {
        return it->second;
    }
    else
    {
        throw std::invalid_argument("Cannot find info for symbol " + symbol->to_string());
    }
}

const SymbolInfo &ReferenceDataReader::GetSymbolInfoFromSymbol(const std::string symbol_str) const
{
    const auto symbol = symbol_manager_->GetSymbolByString(symbol_str);
    if (auto it = symbol_to_info_map_.find(symbol); it != symbol_to_info_map_.end())
    {
        return it->second;
    }
    else
    {
        throw std::invalid_argument("Cannot find info for symbol " + symbol_str);
    }
}

int ReferenceDataReader::GetRetrievingDays() const
{
    return retrieving_days_;
}

int ReferenceDataReader::GetLeastTradeCount() const
{
    return least_trade_count_;
}

const Date &ReferenceDataReader::GetFirstDate() const
{
    return first_date_;
}
}  // namespace alphaone
