#ifndef _MARK_DATA_PROVIDER_TWSE_DATA_FILE_H_
#define _MARK_DATA_PROVIDER_TWSE_DATA_FILE_H_

#include "boost/bimap.hpp"
#include "infrastructure/base/MarketDataProvider.h"
#include "infrastructure/common/file/TWSEDataFileReader.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/datasource/MarketDataSource.h"

namespace alphaone
{
template <typename L, typename R>
boost::bimap<L, R> make_bimap(std::initializer_list<typename boost::bimap<L, R>::value_type> list)
{
    return boost::bimap<L, R>(list.begin(), list.end());
}

class MarketDataProvider_TWSEDataFile : public MarketDataProvider
{
  public:
    MarketDataProvider_TWSEDataFile(DataSourceID data_source_id);
    ~MarketDataProvider_TWSEDataFile();

    void AddReader(const std::filesystem::path &fileroot, const std::string &pid);

    void            Process(const Timestamp &event_loop_time);
    const Timestamp PeekTimestamp();

    DataSourceType GetType() const
    {
        return DataSourceType::MarketByOrder;
    }

    bool IsInitialized() const;
    bool IsFinished() const;

    ProviderID GetProviderID() const;

    static ExternalOrderId GetExternalOrderId(const std::string_view &broker_code,
                                              const std::string_view &order_number);
    static std::string     GetBrokerOrderNumber(ExternalOrderId id);

  private:
    void Parse(const TWSEDataReport &report, const bool check_delete_after_trade = true);
    void Notify(std::vector<MarketDataListener *> &list, MarketDataMessage *msg, void *raw_packet);

    MarketDataMessage                                     marketdata_message_;
    MarketDataSource *                                    market_data_source_;
    DataSourceID                                          data_source_id_;
    bool                                                  is_initialized_;
    bool                                                  is_finished_;
    std::unordered_map<std::string, TWSEDataFileReader *> pid_to_reader_map_;
    TWSEDataReport                                        delete_after_trade_report_;
    Timestamp                                             closing_timestamp_;
};

}  // namespace alphaone

#endif
