#include "MarketDataProvider_TWSEDataFile.h"

#include "infrastructure/common/message/TWSEDataFileFormat.h"
#include "infrastructure/common/util/String.h"

namespace alphaone
{
static const boost::bimap<char, uint64_t> ORDER_NUMBER_MAPPING = make_bimap<char, uint64_t>(
    {{'0', 0},  {'1', 1},  {'2', 2},  {'3', 3},  {'4', 4},  {'5', 5},  {'6', 6},  {'7', 7},
     {'8', 8},  {'9', 9},  {'A', 10}, {'B', 11}, {'C', 12}, {'D', 13}, {'E', 14}, {'F', 15},
     {'G', 16}, {'H', 17}, {'I', 18}, {'J', 19}, {'K', 20}, {'L', 21}, {'M', 22}, {'N', 23},
     {'O', 24}, {'P', 25}, {'Q', 26}, {'R', 27}, {'S', 28}, {'T', 29}, {'U', 30}, {'V', 31},
     {'W', 32}, {'X', 33}, {'Y', 34}, {'Z', 35}, {'a', 36}, {'b', 37}, {'c', 38}, {'d', 39},
     {'e', 40}, {'f', 41}, {'g', 42}, {'h', 43}, {'i', 44}, {'j', 45}, {'k', 46}, {'l', 47},
     {'m', 48}, {'n', 49}, {'o', 50}, {'p', 51}, {'q', 52}, {'r', 53}, {'s', 54}, {'t', 55},
     {'u', 56}, {'v', 57}, {'w', 58}, {'x', 59}, {'y', 60}, {'z', 61}, {' ', 62}});

static const uint64_t ORDER_NUMBER_TRANSFORMER[9] = {1UL,
                                                     63UL,
                                                     63UL * 63UL,
                                                     63UL * 63UL * 63UL,
                                                     63UL * 63UL * 63UL * 63UL,
                                                     63UL * 63UL * 63UL * 63UL * 63UL,
                                                     63UL * 63UL * 63UL * 63UL * 63UL * 63UL,
                                                     63UL * 63UL * 63UL * 63UL * 63UL * 63UL * 63UL,
                                                     63UL * 63UL * 63UL * 63UL * 63UL * 63UL *
                                                         63UL * 63UL};

MarketDataProvider_TWSEDataFile::MarketDataProvider_TWSEDataFile(DataSourceID data_source_id)
    : MarketDataProvider{DataSourceType::MarketByOrder}
    , marketdata_message_{DataSourceType::MarketByOrder}
    , data_source_id_{data_source_id}
    , is_initialized_{false}
    , is_finished_{false}
    , closing_timestamp_{Timestamp::invalid()}
{
    market_data_source_                         = GetMarketDataSource(data_source_id_);
    marketdata_message_.trade.is_not_duplicate_ = true;
    marketdata_message_.provider_id             = ProviderID::TWSE_DATA_FILE;
    delete_after_trade_report_.Type             = TWSEDataReportType::None;
}

MarketDataProvider_TWSEDataFile::~MarketDataProvider_TWSEDataFile()
{
    for (auto &[pid, reader] : pid_to_reader_map_)
    {
        if (reader != nullptr)
        {
            delete reader;
            reader = nullptr;
        }
    }
}

void MarketDataProvider_TWSEDataFile::AddReader(const std::filesystem::path &fileroot,
                                                const std::string &          pid)
{
    std::string filename_order{"odr" + pid};
    std::string filename_match{"mth" + pid};

    if (auto it = pid_to_reader_map_.find(pid);
        it != pid_to_reader_map_.end() && it->second != nullptr)
    {
        delete it->second;
    }
    pid_to_reader_map_[pid] =
        new TWSEDataFileReader(fileroot / filename_order, fileroot / filename_match);


    is_initialized_ = true;
}

void MarketDataProvider_TWSEDataFile::Process(const Timestamp &event_loop_time)
{
    if (is_initialized_)
    {
        if (event_loop_time < PeekTimestamp())
        {
            return;
        }

        bool has_next = false;
        for (const auto &[pid, reader] : pid_to_reader_map_)
        {
            if (reader->ReadNext())
            {
                Parse(reader->Get());
                has_next = true;
            }
        }
        if (not has_next)
        {
            is_finished_ = true;
        }
    }
}

void MarketDataProvider_TWSEDataFile::Parse(const TWSEDataReport &report,
                                            const bool            check_delete_after_trade)
{
    if (check_delete_after_trade && delete_after_trade_report_.Type != TWSEDataReportType::None &&
        report.Type != TWSEDataReportType::Trade)
    {
        // delete IOC or FOK
        Parse(delete_after_trade_report_, false);
        delete_after_trade_report_.Type = TWSEDataReportType::None;
    }

    auto &symbol_str = report.SecuritiesCode;
    auto  t = market_data_source_->GetMarketDataListenerNode(symbol_str.data(), symbol_str.size());

    if (BRANCH_LIKELY(t != nullptr))
    {
        const auto symbol = std::get<0>(*t);

        marketdata_message_.symbol          = std::get<0>(*t);
        marketdata_message_.provider_time   = report.Time;
        marketdata_message_.exchange_time   = report.Time;
        marketdata_message_.sequence_number = 0;

        marketdata_message_.mbo.is_packet_end = true;
        marketdata_message_.mbo.nord          = 1;
        marketdata_message_.mbo.order_id =
            GetExternalOrderId(report.BrokerCode, report.OrderNumber);
        marketdata_message_.mbo.price =
            static_cast<double>(static_cast<int64_t>(report.Price * symbol->GetDecimalConverter()));

        marketdata_message_.mbo.side = report.BuySell;
        if (report.Type == TWSEDataReportType::Add)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_Add;
            marketdata_message_.mbo.qty = report.Volume * 0.001;
        }
        else if (report.Type == TWSEDataReportType::AdjustVolume)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_ModifyWithQty;
            marketdata_message_.mbo.qty = report.Volume * 0.001;
        }
        else if (report.Type == TWSEDataReportType::AdjustPrice)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_ModifyWithPrice;
            marketdata_message_.mbo.qty = report.Volume * 0.001;
        }
        else if (report.Type == TWSEDataReportType::Cancel)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_Delete;
            marketdata_message_.mbo.qty = std::abs(report.Volume) * 0.001;
        }
        else if (report.Type == TWSEDataReportType::Trade)
        {
            marketdata_message_.market_data_message_type =
                MarketDataMessageType::MarketDataMessageType_Trade;
            marketdata_message_.trade.order_id              = marketdata_message_.mbo.order_id;
            marketdata_message_.trade.counterparty_order_id = 0;
            marketdata_message_.trade.price                 = marketdata_message_.mbo.price;
            marketdata_message_.trade.qty                   = report.Volume * 0.001;
            marketdata_message_.trade.side                  = marketdata_message_.mbo.side;
            marketdata_message_.trade.is_packet_end         = true;
        }

        if (report.TradeTypeCode != TWSEDataReportTradeTypeCode::Odd)
        {
            if (!closing_timestamp_.is_valid())
            {
                closing_timestamp_ = Timestamp::from_date_time(
                    marketdata_message_.provider_time.to_date(), "14:00:00.000000000");
            }

            if (marketdata_message_.provider_time < closing_timestamp_)
            {
                Notify(std::get<1>(*t), &marketdata_message_,
                       const_cast<TWSEDataReport *>(&report));
            }
        }

        if (report.TimeRestriction != TWSEDataReportTimeRestriction::ROD &&
            report.Type == TWSEDataReportType::Add)
        {
            // save IOC or FOK, and delete after trade packet
            delete_after_trade_report_      = report;
            delete_after_trade_report_.Type = TWSEDataReportType::Cancel;
        }
    }
}

void MarketDataProvider_TWSEDataFile::Notify(std::vector<MarketDataListener *> &list,
                                             MarketDataMessage *msg, void *raw_packet)
{
    for (auto &l : list)
    {
        l->OnMarketDataMessage(msg, raw_packet);
    }
}

const Timestamp MarketDataProvider_TWSEDataFile::PeekTimestamp()
{
    for (const auto &[pid, reader] : pid_to_reader_map_)
    {
        if (reader->ReadNext())
        {
            return reader->PeekTimestamp();
        }
    }
    return Timestamp::invalid();
}

ProviderID MarketDataProvider_TWSEDataFile::GetProviderID() const
{
    return ProviderID::TWSE_DATA_FILE;
}

ExternalOrderId
MarketDataProvider_TWSEDataFile::GetExternalOrderId(const std::string_view &broker_code,
                                                    const std::string_view &order_number)
{
    return ORDER_NUMBER_MAPPING.left.at(broker_code[0]) * ORDER_NUMBER_TRANSFORMER[0] +
           ORDER_NUMBER_MAPPING.left.at(broker_code[1]) * ORDER_NUMBER_TRANSFORMER[1] +
           ORDER_NUMBER_MAPPING.left.at(broker_code[2]) * ORDER_NUMBER_TRANSFORMER[2] +
           ORDER_NUMBER_MAPPING.left.at(broker_code[3]) * ORDER_NUMBER_TRANSFORMER[3] +
           ORDER_NUMBER_MAPPING.left.at(order_number[0]) * ORDER_NUMBER_TRANSFORMER[4] +
           ORDER_NUMBER_MAPPING.left.at(order_number[1]) * ORDER_NUMBER_TRANSFORMER[5] +
           ORDER_NUMBER_MAPPING.left.at(order_number[2]) * ORDER_NUMBER_TRANSFORMER[6] +
           ORDER_NUMBER_MAPPING.left.at(order_number[3]) * ORDER_NUMBER_TRANSFORMER[7] +
           ORDER_NUMBER_MAPPING.left.at(order_number[4]) * ORDER_NUMBER_TRANSFORMER[8];
}

std::string MarketDataProvider_TWSEDataFile::GetBrokerOrderNumber(ExternalOrderId id)
{
    std::string result(9, ' ');
    for (size_t i{0}; i < 9; ++i)
    {
        result[i] = ORDER_NUMBER_MAPPING.right.at(id % 63);
        id /= 63;
    }
    return result;
}

bool MarketDataProvider_TWSEDataFile::IsInitialized() const
{
    return is_initialized_;
}

bool MarketDataProvider_TWSEDataFile::IsFinished() const
{
    return is_finished_;
}

}  // namespace alphaone
