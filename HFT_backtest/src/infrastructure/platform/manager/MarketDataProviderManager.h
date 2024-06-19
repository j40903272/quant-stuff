#ifndef _MARKET_DATA_PROVIDER_MANAGER_H_
#define _MARKET_DATA_PROVIDER_MANAGER_H_

#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/PacketLog.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_AlphaOne.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_TWSE.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_TWSEDataFile.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_Taifex.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_BinancePerp.h"
#include "infrastructure/platform/manager/ObjectManager.h"

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace alphaone
{
class MarketDataProviderManager
{
  public:
    MarketDataProviderManager(const EngineEventLoopType event_loop_type,
                              const ObjectManager *     obj_manager);
    ~MarketDataProviderManager();

    void             AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener);
    const Timestamp  PeekTimestamp();
    void             Process(const Timestamp &event_loop_time);
    void             SortMarketDataListener();
    void             SetPacketLogStruct(PacketLogStruct *packet_log_struct);
    PacketLogStruct *GetPacketLogStruct();

  private:
    static const inline std::unordered_set<DataSourceID> alphaone_datasource_id_{
        DataSourceID::TAIFEX_FUTURE, DataSourceID::TAIFEX_OPTION,      DataSourceID::TSE,
        DataSourceID::OTC,           DataSourceID::SGX_FUTURE,         DataSourceID::CME,
        DataSourceID::TPRICE,        DataSourceID::SGX_REALTIME_FUTURE, DataSourceID::BINANCE_PERP};

    const EngineEventLoopType event_loop_type_;
    const ObjectManager *     object_manager_;
    PacketLogStruct *         packet_log_struct_;
    bool                      isEnablePacketLog_;

    std::unordered_map<DataSourceID, MarketDataProvider *> data_source_id_to_provider_;
    std::unordered_map<ProviderID, MarketDataProvider *>   provider_id_to_provider_;

    std::vector<MarketDataProvider *> provider_list_;

    void AddProductionProvider(const Symbol *symbol, MarketDataListener *listener);
    void AddSimulationProvider(const Symbol *symbol, MarketDataListener *listener);

    MarketDataProvider *CreateAlphaOneProvider(const std::filesystem::path &root,
                                               const nlohmann::json &node, DataSourceID id);
    MarketDataProvider *CreateBinancePerpProvider(const std::filesystem::path &root,
                                                  const nlohmann::json &node, DataSourceID id);
    MarketDataProvider *CreateTWSEDataFileProvider(const std::filesystem::path &root,
                                                   const std::string &pid, DataSourceID id);
};
}  // namespace alphaone
#endif
