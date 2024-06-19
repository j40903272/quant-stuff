#ifndef _OBJECTMANAGER_H
#define _OBJECTMANAGER_H

#include "infrastructure/common/configuration/GlobalConfiguration.h"
#include "infrastructure/platform/manager/SymbolManager.h"
#include "infrastructure/platform/reader/ReferenceDataReader.h"
#include "infrastructure/platform/reader/SharesReader.h"

namespace alphaone
{
class ObjectManager
{
  public:
    ObjectManager() : symbol_manager_{nullptr}, reference_data_{nullptr}, shares_reader_{nullptr}
    {
    }

    template <typename Config>
    ObjectManager(const Date &date, const Config &config)
        : symbol_manager_{nullptr}, reference_data_{nullptr}, shares_reader_{nullptr}
    {
        global_configurations_.push_back(new GlobalConfiguration{});
        auto gc = global_configurations_.back();
        gc->Load(config);
        std::cout << 111 << std ::endl;
        Init(date, gc);
        std::cout << 111 << std ::endl;
    }

    void Init(const Date &date, GlobalConfiguration *gc)
    {
        const auto &g_json = gc->GetJson();
        std::cout << 222 << std ::endl;
        auto        node   = g_json.at("MarketData");
        std::cout << 222 << std ::endl;
        fileroot_ = std::filesystem::path(node["fileroot"].get<std::string>()) / node["location"] / date.to_string_with_dash();
        std::cout << 222 << std ::endl;
        if (g_json.contains("Providers"))
        {
            std::cout << 222 << std ::endl;
            const auto &p_json = g_json["Providers"];
            std::cout << 222 << std ::endl;
            for (const auto &[pkey, pvalue] : p_json.items())
            {
                std::cout << 333 << std ::endl;
                auto sit = string_to_id_.find(pkey);
                if (sit == string_to_id_.end())
                {
                    SPDLOG_WARN("Invalid Provider {}, skip adding path", pkey);
                    continue;
                }
                std::cout << 444 << std ::endl;
                auto marketdata_root = fileroot_;
                if (pvalue.contains("fileroot") && pvalue.contains("location"))
                {
                    marketdata_root = std::filesystem::path{pvalue["fileroot"].get<std::string>()} /
                                      pvalue["location"] / date.to_string_with_dash();
                }

                provider_paths_[sit->second] = {marketdata_root, pvalue};
            }
        }

        std::cout << 666 << std ::endl;
        // set up symbol manager
        symbol_manager_ = new SymbolManager{};
        symbol_manager_->Load(fileroot_ / "symbols.json");
        SPDLOG_INFO("sad");
        // set up reference data reader
        nlohmann::json rn{
            {"retrieving_days", 15}, {"least_trade_count", 60}, {"is_from_today", false}};
        if (g_json.contains("ReferenceData"))
        {
            const auto &rj        = g_json["ReferenceData"];
            rn["retrieving_days"] = rj.value("retrieving_days", rn["retrieving_days"].get<int>());
            rn["least_trade_count"] =
                rj.value("least_trade_count", rn["least_trade_count"].get<int>());
            rn["is_from_today"] = rj.value("is_from_today", rn["is_from_today"].get<bool>());
        }
        std::cout << 878 << std ::endl;
        // if (rn["retrieving_days"].get<int>() > 0)
        //     reference_data_ = new ReferenceDataReader{symbol_manager_, rn};

        // // set up shares reader
        // if (g_json.contains("SharesReader"))
        // {
        //     const auto &srj      = g_json["SharesReader"];
        //     auto        filename = srj.value("filename", "shares.csv");
        //     auto path      = srj.value("path", fmt::format("{}/{}", fileroot_.string(), filename));
        //     shares_reader_ = new SharesReader(symbol_manager_, path);
        // }
    }

    ObjectManager(const Date &date, const std::vector<std::string> &configs)
        : ObjectManager(date, configs[0])
    {
        for (size_t i = 1; i < configs.size(); ++i)
        {
            global_configurations_.push_back(new GlobalConfiguration{});
            global_configurations_.back()->Load(configs[i]);
        }
    }

    virtual ~ObjectManager()
    {
        if (shares_reader_ != nullptr)
        {
            delete shares_reader_;
            shares_reader_ = nullptr;
        }

        if (reference_data_ != nullptr)
        {
            delete reference_data_;
            reference_data_ = nullptr;
        }

        if (symbol_manager_ != nullptr)
        {
            delete symbol_manager_;
            symbol_manager_ = nullptr;
        }

        for (auto gc : global_configurations_)
        {
            if (gc != nullptr)
            {
                delete gc;
                gc = nullptr;
            }
        }
    }

    virtual SymbolManager *GetSymbolManager() const
    {
        return symbol_manager_;
    }

    virtual GlobalConfiguration *GetGlobalConfiguration() const
    {
        if (global_configurations_.empty())
        {
            return nullptr;
        }
        return global_configurations_.front();
    }


    virtual GlobalConfiguration *GetGlobalConfiguration(size_t config_index) const
    {
        if (config_index >= global_configurations_.size())
        {
            return nullptr;
        }
        return global_configurations_[config_index];
    }

    virtual const std::vector<GlobalConfiguration *> &GetGlobalConfigurations() const
    {
        return global_configurations_;
    }

    virtual const ReferenceDataReader *GetReferenceData() const
    {
        return reference_data_;
    }

    virtual const SharesReader *GetShares() const
    {
        return shares_reader_;
    }

    // root of symbols.json, t.json. reference_data.csv, shares.csv ...
    virtual std::filesystem::path GetFileRoot() const
    {
        return fileroot_;
    }

    virtual std::pair<std::filesystem::path, nlohmann::json> GetMarketDataPath(ProviderID id) const
    {
        auto pit = provider_paths_.find(id);
        if (pit == provider_paths_.end())
            return {"", nlohmann::json{}};

        return pit->second;
    }

    virtual const std::unordered_map<ProviderID, std::pair<std::filesystem::path, nlohmann::json>> &
    GetMarketDataPaths() const
    {
        return provider_paths_;
    }

  private:
    static inline const std::unordered_map<std::string, ProviderID> string_to_id_{
        {"AlphaOne", ProviderID::AlphaOne},
        {"TAIFEX_FUTURE", ProviderID::TAIFEX_FUTURE},
        {"TWSE_DATA_FILE", ProviderID::TWSE_DATA_FILE},
        {"BINANCE_PERP", ProviderID::BINANCE_PERP}};

  protected:
    std::vector<GlobalConfiguration *> global_configurations_;
    SymbolManager *                    symbol_manager_;
    ReferenceDataReader *              reference_data_;
    SharesReader *                     shares_reader_;

    std::filesystem::path fileroot_;

    std::unordered_map<ProviderID, std::pair<std::filesystem::path, nlohmann::json>>
        provider_paths_;
};
}  // namespace alphaone
#endif
