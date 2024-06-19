#ifndef MARKETDATASOURCE_H
#define MARKETDATASOURCE_H

#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/message/MarketDataMessage.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/manager/MultiBookManager.h"

#include <string.h>
#include <tuple>
#include <vector>

namespace alphaone
{
using MarketDataListenerNode =
    std::tuple<const Symbol *, std::vector<MarketDataListener *>, void *>;

class MarketDataSource
{
  public:
    MarketDataSource(DataSourceID data_source_id, DataSourceType data_source_type);
    MarketDataSource(const MarketDataSource &) = delete;
    MarketDataSource &operator=(const MarketDataSource &) = delete;

    ~MarketDataSource();
    void AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener);

    void SortMarketDataListener();

    DataSourceID   GetDataSourceID() const;
    DataSourceType GetType() const;

    bool AddExtraData(const Symbol *symbol, void *data);

    MarketDataListenerNode *GetMarketDataListenerNode(const char *pid, const int len);
    const std::vector<MarketDataListenerNode> &GetMarketDataListenerNodes();

  private:
    std::vector<MarketDataListenerNode> market_data_nodes_;

    DataSourceID   data_source_id_;
    DataSourceType data_source_type_;
};

}  // namespace alphaone

#endif
