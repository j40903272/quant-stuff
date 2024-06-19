#ifndef _REFERENCEDATAREADER_H_
#define _REFERENCEDATAREADER_H_

#include "infrastructure/base/Ohlcv.h"
#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/file/DelimitedFileReader.h"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/symbol/SymbolInfo.h"
#include "infrastructure/common/util/Macro.h"
#include "infrastructure/platform/manager/SymbolManager.h"

#include <filesystem>
#include <unordered_map>

namespace alphaone
{

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)

#define INSERT_INFO(INFO, KEY, SRC) INFO.CONCAT(KEY, _) = SRC.RetrieveDouble(#KEY);
#define INSERT_BY_KEY(KEY) INSERT_INFO(it_pair.first->second, KEY, reader);
#define ADD_INFO(INFO, KEY, SRC) INFO.CONCAT(KEY, _) += SRC.RetrieveDouble(#KEY);
#define ADD_BY_KEY(KEY) ADD_INFO(it_pair.first->second, KEY, reader);

class ReferenceDataReader
{

  public:
    ReferenceDataReader(const SymbolManager *symbol_manager, nlohmann::json node);
    ~ReferenceDataReader();
    void              Read();
    const SymbolInfo &GetSymbolInfoFromSymbol(const Symbol *symbol) const;
    const SymbolInfo &GetSymbolInfoFromSymbol(const std::string symbol_str) const;
    int               GetRetrievingDays() const;
    int               GetLeastTradeCount() const;
    const Date &      GetFirstDate() const;

  private:
    const SymbolManager *                          symbol_manager_;
    const int                                      retrieving_days_;
    const int                                      least_trade_count_;
    const std::string                              file_name_;
    Date                                           first_date_;
    std::filesystem::path                          root_path_;
    std::unordered_map<const Symbol *, SymbolInfo> symbol_to_info_map_;
};

}  // namespace alphaone


#endif
