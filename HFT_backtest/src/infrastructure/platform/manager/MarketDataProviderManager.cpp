#include "MarketDataProviderManager.h"

namespace alphaone
{

MarketDataProviderManager::MarketDataProviderManager(const EngineEventLoopType event_loop_type,
                                                     const ObjectManager *     obj_manager)
    : event_loop_type_{event_loop_type}, object_manager_{obj_manager}, isEnablePacketLog_(false)
{
}

MarketDataProviderManager::~MarketDataProviderManager()
{
    for (auto &provider : provider_list_)
    {
        delete provider;
    }
}

void MarketDataProviderManager::AddMarketDataListener(const Symbol *      symbol,
                                                      MarketDataListener *listener)
{
    if (event_loop_type_ == EngineEventLoopType::Production)
    {
        AddProductionProvider(symbol, listener);
    }
    else if (event_loop_type_ == EngineEventLoopType::Simulation)
    {
        SPDLOG_INFO("AAAAAAAAAAAAAAA");
        SPDLOG_INFO("{} {} {} {}", symbol->GetPid(), symbol->GetSource(), symbol->GetGroup(), symbol->GetDataSourcePid());
        AddSimulationProvider(symbol, listener);
    }
    else
    {
        SPDLOG_ERROR("MarketDataProviderManager EngineEventLoopType Error");
    }
}

void MarketDataProviderManager::AddProductionProvider(const Symbol *      symbol,
                                                      MarketDataListener *listener)
{
    MarketDataProvider *provider = nullptr;

    const auto data_source_id = symbol->GetDataSourceID();

    if (auto dit = data_source_id_to_provider_.find(data_source_id);
        dit == data_source_id_to_provider_.end())
    {
        if (data_source_id == DataSourceID::TAIFEX_FUTURE)
        {
            provider             = new MarketDataProvider_Taifex(data_source_id);
            auto tx_fut_provider = dynamic_cast<MarketDataProvider_Taifex *>(provider);
            tx_fut_provider->Init();
            tx_fut_provider->SetPacketLogStruct(packet_log_struct_);
            provider_id_to_provider_[ProviderID::TAIFEX_FUTURE] = provider;
        }

        if (provider != nullptr)
        {
            data_source_id_to_provider_[data_source_id] = provider;
            provider_list_.push_back(provider);
        }
    }
    else
    {
        provider = dit->second;
    }

    if (provider != nullptr)
    {
        provider->AddMarketDataListener(symbol, listener);
    }
}

void MarketDataProviderManager::AddSimulationProvider(const Symbol *      symbol,
                                                      MarketDataListener *listener)
{
    const auto &data_source_id{symbol->GetDataSourceID()};
    SPDLOG_INFO("yo bitch123");
    if (data_source_id == DataSourceID::TWSE_DATA_FILE || data_source_id == DataSourceID::TSE)
    {
        const auto &[path, pjson] = object_manager_->GetMarketDataPath(ProviderID::TWSE_DATA_FILE);
        if (!path.empty())
        {
            auto provider = CreateTWSEDataFileProvider(path, symbol->GetPid(), data_source_id);
            provider->AddMarketDataListener(symbol, listener);
        }
    }
    SPDLOG_INFO("???? {}", data_source_id == DataSourceID::BINANCE_PERP);
    if (alphaone_datasource_id_.find(data_source_id) != alphaone_datasource_id_.end())
    {
        SPDLOG_INFO("yo bitch here");
        const auto &[path, pjson] = object_manager_->GetMarketDataPath(ProviderID::AlphaOne);
        if (!path.empty() && !pjson.empty())
        {
            auto provider = CreateAlphaOneProvider(path, pjson, data_source_id);
            provider->AddMarketDataListener(symbol, listener);
        }
        
        const auto &[path2, pjson2] = object_manager_->GetMarketDataPath(ProviderID::BINANCE_PERP);
        if (!path2.empty() && !pjson2.empty())
        {
            SPDLOG_INFO("AddSimProvider: {}", data_source_id);
            auto provider = CreateBinancePerpProvider(path2, pjson2, data_source_id);
            provider->AddMarketDataListener(symbol, listener);
        }
        
    }
}

MarketDataProvider *
MarketDataProviderManager::CreateAlphaOneProvider(const std::filesystem::path &root,
                                                  const nlohmann::json &node, DataSourceID id)
{
    auto pit = provider_id_to_provider_.find(ProviderID::AlphaOne);
    if (pit != provider_id_to_provider_.end())
        return pit->second;

    auto provider = new MarketDataProvider_AlphaOne();
    auto filename = node.value("filename", "marketdata.bin");
    auto filepath = root / filename;
    if (std::filesystem::exists(filepath))
    {
        provider->Init(filepath);
    }
    else
    {
        SPDLOG_INFO("[{}] {} does not exist", __func__, filepath);
        filepath = root / "marketdata.bin";
        if (std::filesystem::exists(filepath))
        {
            SPDLOG_INFO("[{}] use default file {} instead", __func__, filepath);
            provider->Init(filepath);
        }
        else
        {
            throw std::invalid_argument(
                fmt::format("Cannot find {} or marketdata.bin under {}", filename, root));
        }
    }

    data_source_id_to_provider_[id]                = provider;
    provider_id_to_provider_[ProviderID::AlphaOne] = provider;
    provider_list_.push_back(provider);
    return dynamic_cast<MarketDataProvider *>(provider);
}

MarketDataProvider *
MarketDataProviderManager::CreateBinancePerpProvider(const std::filesystem::path &root,
                                                  const nlohmann::json &node, DataSourceID id)
{
    SPDLOG_INFO("[{}]", __func__);
    auto pit = provider_id_to_provider_.find(ProviderID::BINANCE_PERP);
    if (pit != provider_id_to_provider_.end())
        return pit->second;

    auto provider = new MarketDataProvider_BinancePerp();
    auto filename = node.value("filename", "a.txt");
    auto filepath = root / filename;
    if (std::filesystem::exists(filepath))
    {
        provider->Init(filepath);
    }
    else
    {
        SPDLOG_INFO("[{}] {} does not exist", __func__, filepath);
        filepath = root / "a.txt";
        if (std::filesystem::exists(filepath))
        {
            SPDLOG_INFO("[{}] use default file {} instead", __func__, filepath);
            provider->Init(filepath);
        }
        else
        {
            throw std::invalid_argument(
                fmt::format("Cannot find {} or a.txt under {}", filename, root));
        }
    }

    data_source_id_to_provider_[id]                = provider;
    provider_id_to_provider_[ProviderID::BINANCE_PERP] = provider;
    provider_list_.push_back(provider);
    SPDLOG_INFO("[{}] provider!!", __func__);
    return dynamic_cast<MarketDataProvider *>(provider);
}

MarketDataProvider *
MarketDataProviderManager::CreateTWSEDataFileProvider(const std::filesystem::path &root,
                                                      const std::string &pid, DataSourceID id)
{
    SPDLOG_INFO("[{}] {}", __func__, object_manager_->GetSymbolManager()->GetDate());
    MarketDataProvider_TWSEDataFile *provider{nullptr};
    auto pit = provider_id_to_provider_.find(ProviderID::TWSE_DATA_FILE);
    if (pit != provider_id_to_provider_.end())
        provider = dynamic_cast<MarketDataProvider_TWSEDataFile *>(pit->second);
    else
        provider = new MarketDataProvider_TWSEDataFile(id);

    if (!provider->IsInitialized())
    {
        data_source_id_to_provider_[id]                      = provider;
        provider_id_to_provider_[ProviderID::TWSE_DATA_FILE] = provider;
        provider_list_.push_back(provider);
    }

    provider->AddReader(root, pid);
    return dynamic_cast<MarketDataProvider *>(provider);
}

void MarketDataProviderManager::Process(const Timestamp &event_loop_time)
{
    for (auto provider : provider_list_)
    {
        if (isEnablePacketLog_)
        {
            packet_log_struct_->Type      = (int)PacketLogType::INVALID;
            packet_log_struct_->SeqNum    = 101;
            packet_log_struct_->LoopCount = 0;
        }
        provider->Process(event_loop_time);
    }

    if (isEnablePacketLog_)
    {
        packet_log_struct_->Type      = (int)PacketLogType::INVALID;
        packet_log_struct_->SeqNum    = 101;
        packet_log_struct_->LoopCount = 0;
    }
}

const Timestamp MarketDataProviderManager::PeekTimestamp()
{
    std::cout<< "WTFFFF" << std::endl;
    Timestamp earliest_timestamp{Timestamp::max_time()};
    SPDLOG_INFO("length of provider_list_ {}", provider_list_.size());
    for (auto provider : provider_list_)
    {
        SPDLOG_INFO("provider: {}", provider->GetProviderID());
        // std::cout << "provider: "<< (char)provider->GetProviderID() << std::endl;
        Timestamp peeked_timestamp{provider->PeekTimestamp()};
        std::cout << "peeked_timestamp: "<< peeked_timestamp << std::endl;
        if (peeked_timestamp.is_valid() && peeked_timestamp < earliest_timestamp)
        {
            earliest_timestamp = peeked_timestamp;
        }
    }
    return earliest_timestamp;
}

void MarketDataProviderManager::SortMarketDataListener()
{
    for (auto provider : provider_list_)
    {
        provider->SortMarketDataListener();
    }
}

void MarketDataProviderManager::SetPacketLogStruct(PacketLogStruct *packet_log_struct)
{
    packet_log_struct_ = packet_log_struct;
    if (packet_log_struct_ != nullptr)
    {
        isEnablePacketLog_ = true;
    }
}


PacketLogStruct *MarketDataProviderManager::GetPacketLogStruct()
{
    return packet_log_struct_;
}

}  // namespace alphaone
