#include "MarketDataSource.h"

#include "infrastructure/base/Book.h"

namespace alphaone
{

MarketDataSource::MarketDataSource(DataSourceID data_source_id, DataSourceType data_source_type)
    : data_source_id_{data_source_id}, data_source_type_{data_source_type}
{
}

MarketDataSource::~MarketDataSource()
{
}

void MarketDataSource::AddMarketDataListener(const Symbol *symbol, MarketDataListener *listener)
{
    SPDLOG_INFO("am i here?2 symbol->GetDataSourceID():{}", symbol->GetDataSourceID());
    if (symbol->GetDataSourceID() != data_source_id_)
    {
        return;
    }

    bool is_symbol_found{false};
    for (auto &node : market_data_nodes_)
    {
        if (std::get<0>(node) == symbol)
        {
            SPDLOG_INFO("is_symbol_found");
            std::vector<MarketDataListener *> &listeners{std::get<1>(node)};
            if (std::find(listeners.begin(), listeners.end(), listener) == listeners.end())
            {
                listeners.push_back(listener);
            }
            is_symbol_found = true;
        }
    }

    if (!is_symbol_found)
    {
        SPDLOG_INFO("before size of market_data_nodes_: {}", market_data_nodes_.size());
        market_data_nodes_.push_back(
            std::make_tuple(symbol, std::vector<MarketDataListener *>({listener}), nullptr));
        SPDLOG_INFO("after size of market_data_nodes_: {}", market_data_nodes_.size());
    }
}

bool MarketDataSource::AddExtraData(const Symbol *symbol, void *data)
{
    for (auto &node : market_data_nodes_)
    {
        if (std::get<0>(node) == symbol)
        {
            if (std::get<2>(node) == nullptr)
            {
                std::get<2>(node) = data;
                return true;
            }
        }
    }
    return false;
}

MarketDataListenerNode *MarketDataSource::GetMarketDataListenerNode(const char *pid, const int len)
{
    long int head{0};
    SPDLOG_INFO("size of market_data_nodes_: {}", market_data_nodes_.size());
    long int tail{static_cast<long int>(market_data_nodes_.size() - 1)};
    SPDLOG_INFO("head: {}, tail: {}", head, tail);
    while (head <= tail)
    {
        long int mid{(head + tail) / 2};

        const Symbol *symbol{std::get<0>(market_data_nodes_[mid])};

        // SPDLOG_INFO("pid: {}., symbol: {}.", pid, symbol->GetDataSourcePid().c_str());
        // SPDLOG_INFO("equal? {}", pid == symbol->GetDataSourcePid().c_str());
        // SPDLOG_INFO("equal? {}", pid == symbol->GetDataSourcePid());
        // SPDLOG_INFO("len? {}", len);
        // for (size_t i = 0; i < (size_t)len; ++i) {
        //     char charPid = pid[i];
        //     char charDataSourcePid = symbol->GetDataSourcePid().c_str()[i];
        //     int asciiDiff = static_cast<int>(charPid) - static_cast<int>(charDataSourcePid);

        //     std::cout << "Position " << i << ": "
        //             << "pid = '" << charPid << "' (" << static_cast<int>(charPid) << "), "
        //             << "dataSourcePid = '" << charDataSourcePid << "' (" << static_cast<int>(charDataSourcePid) << "), "
        //             << "Difference = " << asciiDiff << "\n";
        // }
        auto length = std::min(len, static_cast<int>(symbol->GetDataSourcePid().size()));
        int ret{memcmp(pid, symbol->GetDataSourcePid().c_str(), length)};
        SPDLOG_INFO("ret: {}", ret);
        ret = 0;
        if (ret > 0)
        {
            head = mid + 1;
        }
        else if (ret < 0)
        {
            tail = mid - 1;
        }
        else
        {
            return &market_data_nodes_[mid];
        }
    }
    return nullptr;
}

void MarketDataSource::SortMarketDataListener()
{
    SPDLOG_INFO("before sort MarketDataSource::SortMarketDataListener() market_data_nodes_.size(): {}", market_data_nodes_.size());
    std::sort(market_data_nodes_.begin(), market_data_nodes_.end(),
              [](std::tuple<const Symbol *, std::vector<MarketDataListener *>, void *> &p1,
                 std::tuple<const Symbol *, std::vector<MarketDataListener *>, void *> &p2) -> bool
              {
                  const std::string &pid1 = std::get<0>(p1)->GetDataSourcePid();
                  const std::string &pid2 = std::get<0>(p2)->GetDataSourcePid();

                  int ret = memcmp(pid1.c_str(), pid2.c_str(), pid1.size());

                  return ret < 0;
              });
    SPDLOG_INFO("after sort MarketDataSource::SortMarketDataListener() market_data_nodes_.size(): {}", market_data_nodes_.size());
}

DataSourceID MarketDataSource::GetDataSourceID() const
{
    SPDLOG_INFO("am i here?3");
    return data_source_id_;
}

const std::vector<MarketDataListenerNode> &MarketDataSource::GetMarketDataListenerNodes()
{
    SPDLOG_INFO("size {}", market_data_nodes_.size());
    return market_data_nodes_;
}

DataSourceType MarketDataSource::GetType() const
{
    return data_source_type_;
}

}  // namespace alphaone
