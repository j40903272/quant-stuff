#ifndef _MARKETDATAREADER_H
#define _MARKETDATAREADER_H

#include "infrastructure/base/MarketDataListener.h"
#include "infrastructure/base/MarketDataProvider.h"
#include "infrastructure/common/message/TAIFEX.h"
#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_AlphaOne.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_TWSEDataFile.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_Taifex.h"
#include "infrastructure/platform/dataprovider/MarketDataProvider_BinancePerp.h"
#include "infrastructure/platform/manager/ObjectManager.h"

#include <filesystem>
#include <unordered_map>

namespace alphaone
{

class MarketDataReader : public MarketDataListener
{
  public:
    MarketDataReader(const ObjectManager *object_manager);
    virtual ~MarketDataReader();

    virtual void PrepareParser();
    virtual void Start();
    virtual void OnMarketDataMessage(const MarketDataMessage *mdm, void *raw_packet) = 0;

    bool IsParserInit()
    {
        return is_parser_init_;
    }

    void SetInitFlag(const bool &is_init)
    {
        is_init_ = is_init;
    }

    const ObjectManager *GetObjectManager()
    {
        return object_manager_;
    }

    const SymbolManager *GetSymbolManager()
    {
        return symbol_manager_;
    }

    const GlobalConfiguration *GetGlobalConfiguration()
    {
        return global_config_;
    }

    std::vector<std::pair<MarketDataProvider *, std::string>> &GetProviders()
    {
        return providers_;
    }

    std::vector<const Symbol *> &GetSymbols()
    {
        return symbols_;
    }

    int GetSymbolId(const Symbol *symbol)
    {
        if (auto it = symbol_to_symbol_id_map_.find(symbol); it != symbol_to_symbol_id_map_.end())
        {
            return it->second;
        }
        return -1;
    }

    int GetCurrentProviderId()
    {
        return current_provider_id_;
    }

    const Date &GetDate()
    {
        return date_;
    }

  protected:
    std::string delimiter_;
    void        LoadSymbols();
    template <typename Arg, typename... Args>
    void Output(std::fstream *file, bool is_end, Arg &&arg, Args &&...args)
    {
        *file << std::forward<Arg>(arg);
        using expander = int[];
        (void)expander{(*file << delimiter_ << std::forward<Args>(args), void(), 0)...};
        if (is_end)
            *file << "\n";
    }

  private:
    const ObjectManager *                   object_manager_;
    const GlobalConfiguration *             global_config_;
    const SymbolManager *                   symbol_manager_;
    const Date                              date_;
    bool                                    is_parser_init_;
    bool                                    is_init_;
    int                                     current_provider_id_;
    std::unordered_map<const Symbol *, int> symbol_to_symbol_id_map_;
    std::vector<const Symbol *>             symbols_;

    std::vector<std::pair<MarketDataProvider *, std::string>> providers_;
};


}  // namespace alphaone
#endif
