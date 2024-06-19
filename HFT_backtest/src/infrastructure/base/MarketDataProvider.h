#ifndef _MARKET_DATA_PROVIDER_H_
#define _MARKET_DATA_PROVIDER_H_

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/datasource/MarketDataSource.h"

#include <string>
#include <unordered_map>

namespace alphaone
{
class MarketDataProvider
{
  public:
    MarketDataProvider(DataSourceType data_source_type);
    virtual ~MarketDataProvider();

    virtual void            Process(const Timestamp &event_loop_time) = 0;
    virtual const Timestamp PeekTimestamp()                           = 0;
    virtual bool            IsValidDate(const DataSourceID &data_source_id, const Date &date)
    {
        return true;
    }
    virtual ProviderID GetProviderID() const = 0;
    virtual bool       IsFinished() const
    {
        return false;
    }
    virtual void AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener);
    virtual bool AddExtraData(const Symbol *symbol, void *data);
    void         SortMarketDataListener();

  protected:
    MarketDataSource *GetMarketDataSource(DataSourceID data_source_id);

  private:
    std::unordered_map<DataSourceID, MarketDataSource *> data_sources_;
    DataSourceType                                       data_source_type_;
};
}  // namespace alphaone
#endif
