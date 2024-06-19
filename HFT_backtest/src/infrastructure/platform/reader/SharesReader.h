#ifndef _SHARESREADER_H_
#define _SHARESREADER_H_

#include "infrastructure/common/file/DelimitedFileReader.h"
#include "infrastructure/common/symbol/SymbolInfo.h"
#include "infrastructure/platform/manager/SymbolManager.h"

#include <filesystem>
#include <unordered_map>

namespace alphaone
{

class SharesReader
{
  public:
    SharesReader(const SymbolManager *sm, const std::filesystem::path csv_path);
    ~SharesReader();

    const ShareInfo &GetShareInfo(const Symbol *symbol) const;

    const std::vector<ShareInfo> &GetShareInfos() const;

  private:
    const Symbol *GetSymbol(const std::string &pid);

    const SymbolManager *sm_;
    DelimitedFileReader  reader_;

    std::vector<ShareInfo> share_infos_;

    std::unordered_map<const Symbol *, size_t> symbol_to_info_id_;
};

}  // namespace alphaone


#endif