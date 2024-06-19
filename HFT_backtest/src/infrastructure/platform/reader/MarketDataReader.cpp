#include "MarketDataReader.h"

#include "infrastructure/common/util/String.h"

namespace alphaone
{

MarketDataReader::MarketDataReader(const ObjectManager *object_manager)
    : delimiter_{","}
    , object_manager_{object_manager}
    , global_config_{object_manager->GetGlobalConfiguration()}
    , symbol_manager_{object_manager->GetSymbolManager()}
    , date_{symbol_manager_->GetDate()}
    , is_parser_init_{false}
    , is_init_{false}
    , current_provider_id_{0}

{
    LoadSymbols();
}

MarketDataReader::~MarketDataReader()
{
    for (auto &p : providers_)
    {
        delete p.first;
    }
}

void MarketDataReader::LoadSymbols()
{
    if (global_config_->GetJson().contains("Universe"))
    {
        int         symbol_id{0};
        const auto &symbol_list = global_config_->GetJson()["Universe"];
        const auto &size        = static_cast<int>(symbol_list.size());
        for (const auto &symbol_json : symbol_list)
        {
            const auto &symbol_str = symbol_json.get<std::string>();
            const auto  symbol     = symbol_manager_->GetSymbolByString(symbol_str);
            if (!symbol)
            {
                SPDLOG_WARN("Skip Invalid symbol_str = {}", symbol_str);
                continue;
            }
            if (symbol_to_symbol_id_map_.insert({symbol, symbol_id}).second)
            {
                symbols_.push_back(symbol);
                ++symbol_id;
            }
        }
    }
    else
    {
        int symbol_id{0};
        // no specify symbol thus iterate all source map
        for (const auto &source_pair : symbol_manager_->GetPidMap())
        {
            for (const auto &type_pair : source_pair.second)
            {
                for (const auto &pid_pair : type_pair.second)
                {
                    if (symbol_to_symbol_id_map_.insert({pid_pair.second, symbol_id}).second)
                    {
                        symbols_.push_back(pid_pair.second);
                        ++symbol_id;
                    }
                }
            }
        }
    }
}

void MarketDataReader::PrepareParser()
{
    for (const auto &[id, root_json_pair] : object_manager_->GetMarketDataPaths())
    {
        if (id == ProviderID::AlphaOne)
        {
            auto p                    = new MarketDataProvider_AlphaOne();
            const auto &[path, pjson] = root_json_pair;
            p->Init(path / pjson.value("filename", "marketdata.bin"));
            providers_.push_back({p, EnumToString::ToString(id)});
        }
        else if (id == ProviderID::TAIFEX_FUTURE)
        {
            auto p = new MarketDataProvider_Taifex(DataSourceID::TAIFEX_FUTURE);
            p->Init();
            providers_.push_back({p, EnumToString::ToString(id)});
        }
        else if (id == ProviderID::BINANCE_PERP)
        {
            auto p                    = new MarketDataProvider_BinancePerp();
            const auto &[path, pjson] = root_json_pair;
            p->Init(path / pjson.value("filename", "marketdata.bin"));
            providers_.push_back({p, EnumToString::ToString(id)});
        }
    }

    is_parser_init_ = true;
}


void MarketDataReader::Start()
{
    if (!is_init_)
        throw std::runtime_error(fmt::format("No init before {}", __func__));

    for (auto &provider : GetProviders())
    {
        for (const auto &s : GetSymbols())
            provider.first->AddMarketDataListener(s, this);

        provider.first->SortMarketDataListener();
        while (!provider.first->IsFinished())
            provider.first->Process(Timestamp::max_time());

        ++current_provider_id_;
    }
}

}  // namespace alphaone
