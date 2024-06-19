#include "MarketDataProvider.h"

namespace alphaone
{

MarketDataProvider::MarketDataProvider(DataSourceType data_source_type)
    : data_source_type_{data_source_type}
{
}

MarketDataProvider::~MarketDataProvider()
{
    for (auto &[data_source_id, market_data_source] : data_sources_)
    {
        delete market_data_source;
    }
}

void MarketDataProvider::AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener)
{
    SPDLOG_INFO("MarketDataProvider::AddMarketDataListener {}", symbol->GetDataSourceID());
    if (auto market_data_source = GetMarketDataSource(symbol->GetDataSourceID());
        market_data_source != nullptr)
    {
        SPDLOG_INFO("market_data_source->AddMarketDataListener");
        market_data_source->AddMarketDataListener(symbol, listener);
    }
}

bool MarketDataProvider::AddExtraData(const Symbol *symbol, void *data)
{
    SPDLOG_INFO("am i here?6");
    if (auto data_source = data_sources_.find(symbol->GetDataSourceID());
        data_source != data_sources_.end())
    {
        return data_source->second->AddExtraData(symbol, data);
    }
    return false;
}

MarketDataSource *MarketDataProvider::GetMarketDataSource(DataSourceID data_source_id)
{
    SPDLOG_INFO("WTFF123FFF data_source_id={}", data_source_id);
    if (auto data_source = data_sources_.find(data_source_id); data_source != data_sources_.end())
    {
        SPDLOG_INFO("find data_source MarketDataProvider::GetMarketDataSource");
        return data_source->second;
    }
    else
    {
        MarketDataSource *market_data_source =
            new MarketDataSource(data_source_id, data_source_type_);

        data_sources_[data_source_id] = market_data_source;
        SPDLOG_INFO("MarketDataProvider::GetMarketDataSource");
        return market_data_source;
    }
    return nullptr;
}

void MarketDataProvider::SortMarketDataListener()
{
    SPDLOG_INFO("data_sources size {}", data_sources_.size());
    for (auto &[id, market_data_source] : data_sources_)
    {
        // SPDLOG_INFO("market_data_source size {}", market_data_source.size());
        market_data_source->SortMarketDataListener();
    }
}

}  // namespace alphaone
