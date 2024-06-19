#include "SharesReader.h"

namespace alphaone
{

SharesReader::SharesReader(const SymbolManager *sm, const std::filesystem::path csv_path)
    : sm_{sm}, reader_{csv_path}
{
    share_infos_.reserve(2048);
    size_t index{0};
    while (reader_.ReadNext())
    {
        const auto &entries = reader_.GetSplitEntries();
        if (entries.size() < 14)
            continue;
        auto        symbol       = GetSymbol(entries[0]);
        const auto &pub_shares   = atol(entries[3].c_str());
        const auto &close_p      = std::stod(entries[4]);
        const auto &ref_open_p   = std::stod(entries[5]);
        const auto &market_cap   = std::stod(entries[6]);
        const auto &curr_weight  = std::stod(entries[7]);
        const auto &sec_class    = atoi(entries[8].c_str());
        const auto &n_sec_class  = atoi(entries[9].c_str());
        const auto &n_pub_shares = atol(entries[10].c_str());
        const auto &n_market_cap = std::stod(entries[11]);
        const auto &next_weight  = std::stod(entries[12]);
        share_infos_.emplace_back(symbol, entries[0], entries[1], entries[2], pub_shares, close_p,
                                  ref_open_p, market_cap, curr_weight, sec_class, n_sec_class,
                                  n_pub_shares, n_market_cap, next_weight, entries[13]);
        symbol_to_info_id_.emplace(symbol, index++);
    }
}

SharesReader::~SharesReader()
{
}

const ShareInfo &SharesReader::GetShareInfo(const Symbol *symbol) const
{
    auto it = symbol_to_info_id_.find(symbol);
    if (it == symbol_to_info_id_.end())
        throw std::invalid_argument(
            fmt::format("Cannot find share info for {}", symbol->to_string()));
    return share_infos_[it->second];
}

const std::vector<ShareInfo> &SharesReader::GetShareInfos() const
{
    return share_infos_;
}

const Symbol *SharesReader::GetSymbol(const std::string &pid)
{
    const Symbol *s{nullptr};
    s = sm_->GetSymbolByPid("TSE", "SECURITY", pid);
    if (s)
        return s;
    s = sm_->GetSymbolByPid("OTC", "SECURITY", pid);
    if (s)
        return s;
    s = sm_->GetSymbolByCPid("TAIFEX", "FUTURE", pid);
    if (s)
        return s;
    return s;
}

}  // namespace alphaone
