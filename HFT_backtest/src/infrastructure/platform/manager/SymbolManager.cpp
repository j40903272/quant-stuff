#include "SymbolManager.h"

#include "infrastructure/common/util/Logger.h"
#include "infrastructure/common/util/String.h"

#include <filesystem>

namespace alphaone
{
SymbolManager::SymbolManager() : file_path_{""}
{
}

SymbolManager::~SymbolManager()
{
    for (auto &symbol : created_symbols_)
    {
        delete symbol;
    }
}

void SymbolManager::Load(const std::filesystem::path &file_path)
{
    if (not std::filesystem::exists(file_path))
    {
        SPDLOG_ERROR("Trying to read symbol information from {} but the file does not exist.",
                     file_path);
        abort();
    }

    nlohmann::json jsonSymbolConfig;
    std::ifstream  fs{file_path};
    fs >> jsonSymbolConfig;
    fs.close();
    file_path_ = file_path;

    std::string updown_str[UpDownDir::UpDown] = {"DownLimit", "UpLimit"};

    // extract date from path
    auto date_str{std::prev(file_path_.parent_path().end())->string()};
    date_str.erase(std::remove(date_str.begin(), date_str.end(), '-'), date_str.end());
    date_ = Date::from_date_str(date_str.c_str());

    auto parse_tick = [&](const nlohmann::json &j, const std::string &k)
    { return std::atof(j[k].get<std::string>().c_str()); };

    for (const auto &[source_k, source_v] : jsonSymbolConfig.items())
    {
        SPDLOG_INFO("{} {}", source_k, source_v);
        for (const auto &[tk, tv] : source_v.items())
        {
            SPDLOG_INFO("{} {}", tk, tv);
            for (const auto &[gk, gv] : tv["Group"].items())
            {
                SPDLOG_INFO("{} {}", gk, gv);
                // parse tick
                std::vector<TickSize> ticksizeList;
                const auto &          arr_TickSize = gv["TickSize"];
                if (arr_TickSize.is_array())
                {
                    for (const auto &[tsk, tsv] : arr_TickSize.items())
                    {
                        ticksizeList.emplace_back(parse_tick(tsv, "Start"), parse_tick(tsv, "End"),
                                                  parse_tick(tsv, "Size"));
                    }
                }

                // parse symbols
                const auto &arr_Symbol = gv["Symbol"];
                if (arr_Symbol.is_array())
                {
                    for (auto &[sk, sv] : arr_Symbol.items())
                    {
                        SPDLOG_INFO("{} {}", sk, sv);
                        auto dp{std::atoi(sv.value("DP", "").c_str())};
                        auto producttype = StringToProductType(sv.value("Type", ""));
                        auto warranttype =
                            static_cast<WarrantType>(std::stoi(sv.value("WarrantType", "999")));
                        auto cpflag = StringToCPFlag(sv.value("CPflag", ""));
                        SPDLOG_INFO(4);
                        std::vector<BookPrice> limits[UpDownDir::UpDown];
                        for (int d = UpDownDir::Down; d < UpDownDir::UpDown; ++d)
                        {
                            if (sv.contains(updown_str[d]) && sv[updown_str[d]].is_array())
                            {
                                for (const auto &[lk, lv] : sv[updown_str[d]].items())
                                {
                                    limits[d].emplace_back(std::stod(lv.get<std::string>()));
                                }
                            }
                            else
                            {
                                limits[d].emplace_back(d ? 999999. : 0.01);
                            }
                        }
                        SPDLOG_INFO(3);
                        auto symbol = new Symbol(
                            source_k, gk, sv.value("Pid", ""), sv.value("CPid", ""),
                            sv.value("Underlying", ""), producttype, warranttype,
                            sv.value("MaturityDate", ""), std::atoi(sv.value("Y", "").c_str()),
                            std::atoi(sv.value("M", "").c_str()),
                            std::atoi(sv.value("W", "").c_str()),
                            std::atof(sv.value("CR", "").c_str()),
                            std::atoi(sv.value("Volume", "").c_str()), cpflag,
                            std::atof(sv.value("Strike", "").c_str()),
                            std::atoi(gv.value("Multiplier", "").c_str()), gv.value("Currency", ""),
                            dp, ticksizeList, std::atof(sv.value("RefPrice", "").c_str()),
                            limits[Down], limits[Up], false, nullptr, nullptr);
                        SPDLOG_INFO(1);
                        created_symbols_.push_back(symbol);
                        SPDLOG_INFO(2);
                        const auto &pid =
                            (symbol->GetPid() == "") ? symbol->GetCPid() : symbol->GetPid();
                        SPDLOG_INFO(5);
                        pid_to_symbol_map_[source_k][tk][pid] = symbol;
                        SPDLOG_INFO(6);
                    }
                }
            }
        }
    }
    // CreateTaifexRollSymbol(jsonSymbolConfig);
    // CreateTPriceSymbol(jsonSymbolConfig);
    // CreateTWSEDataFileSymbol(jsonSymbolConfig);
    SPDLOG_INFO(5);
}

const Symbol *SymbolManager::GetSymbolByPid(const std::string &source, const std::string &type,
                                            const std::string &pid) const
{
    if (auto sit = pid_to_symbol_map_.find(source); sit != pid_to_symbol_map_.end())
    {
        if (auto tit = sit->second.find(type); tit != sit->second.end())
        {
            if (auto pit = tit->second.find(pid); pit != tit->second.end())
            {
                return pit->second;
            }
        }
    }
    return nullptr;
}

const Symbol *SymbolManager::GetSymbolByCPid(const std::string &source, const std::string &type,
                                             const std::string &cpid) const
{
    if (auto sit = cpid_to_symbol_map_.find(source); sit != cpid_to_symbol_map_.end())
    {
        if (auto tit = sit->second.find(type); tit != sit->second.end())
        {
            if (auto pit = tit->second.find(cpid); pit != tit->second.end())
            {
                return pit->second;
            }
        }
    }
    return nullptr;
}

const Symbol *SymbolManager::GetSymbolByString(const std::string &symbol) const
{
    std::unordered_map<std::string, std::string> string_unordered_map{
        SplitIntoUnorderedMap(symbol, '|', '=')};
    std::vector<TickSize> ticksizeList;
    std::vector<BookPrice> limits[UpDownDir::UpDown];
    if (string_unordered_map.find("pid") != string_unordered_map.end())
    {
        auto &source{string_unordered_map["source"]};
        auto &type{string_unordered_map["type"]};
        auto &id{string_unordered_map["pid"]};
        SPDLOG_INFO("{} {} {}", source, type, id);
        return GetSymbolByPid(source, type, id);
    }
    else
    {
        SPDLOG_ERROR("Cannot parse symbol information for symbol={}", symbol);
        abort();
    }
}

std::vector<const Symbol *> SymbolManager::GetSymbolsBySource(const std::string &source,
                                                              const std::string &type) const
{
    std::vector<const Symbol *> ret;
    if (auto sit = pid_to_symbol_map_.find(source); sit != pid_to_symbol_map_.end())
    {
        if (auto tit = sit->second.find(type); tit != sit->second.end())
        {
            for (auto &iter : tit->second)
            {
                ret.push_back(iter.second);
            }
        }
    }
    return ret;
}

std::vector<const Symbol *>
SymbolManager::GetSymbolsBySourceAndGroup(const std::string &source, const std::string &type,
                                          const std::string &group) const
{
    std::vector<const Symbol *> ret;
    if (auto sit = pid_to_symbol_map_.find(source); sit != pid_to_symbol_map_.end())
    {
        if (auto tit = sit->second.find(type); tit != sit->second.end())
        {
            for (auto &iter : tit->second)
            {
                if (iter.second->GetGroup() == group)
                {
                    ret.push_back(iter.second);
                }
            }
        }
    }
    return ret;
}

void SymbolManager::CreateTaifexRollSymbol(const nlohmann::json &config)
{
}

void SymbolManager::CreateTPriceSymbol(const nlohmann::json &config)
{
}

void SymbolManager::CreateTWSEDataFileSymbol(const nlohmann::json &config)
{
    std::vector<TickSize>  ticksizeList;
    std::vector<BookPrice> limits;

    for (const auto &[gk, gv] : config["TSE"]["SECURITY"]["Group"].items())
    {
        auto s = GetSymbolByPid("TSE", "SECURITY", gk);
        if (!s)
        {
            SPDLOG_DEBUG("Cannot find {} symbol {}", "TSE", gk);
            continue;
        }

        auto symbol = new Symbol(
            "TWSE_DATA_FILE", gk, s->GetPid(), s->GetCPid(), "", s->GetProductType(),
            s->GetWarrantType(), s->GetMaturity(), s->GetY(), s->GetM(), s->GetW(), 0, 0,
            s->GetCPFlag(), 0.0, s->GetMultiplier(), s->GetCurrency(), s->GetDecimalPrecision(),
            ticksizeList, s->GetReferencePrice(), limits, limits, false, nullptr, nullptr);
        created_symbols_.push_back(symbol);

        const auto &_pid  = (symbol->GetPid() == "") ? symbol->GetCPid() : symbol->GetPid();
        const auto &_cpid = symbol->GetCPid();

        pid_to_symbol_map_["TWSE_DATA_FILE"]["SECURITY"][_pid] = symbol;
        if (_cpid != "")
        {
            cpid_to_symbol_map_["TWSE_DATA_FILE"]["SECURITY"][_cpid] = symbol;
        }
    }
}

const std::unordered_map<
    std::string, std::unordered_map<std::string, std::unordered_map<std::string, const Symbol *>>> &
SymbolManager::GetPidMap() const
{
    return pid_to_symbol_map_;
}

const std::filesystem::path &SymbolManager::GetJsonPath() const
{
    return file_path_;
}

const Date &SymbolManager::GetDate() const
{
    return date_;
}

const Symbol *SymbolManager::GetOptionPairSymbol(const Symbol *symbol) const
{
    if (symbol->GetProductType() != ProductType::Option ||
        symbol->GetDataSourceID() != DataSourceID::TAIFEX_OPTION)
    {
        return nullptr;
    }
    auto pid{symbol->GetPid()};
    auto m{pid[8]};
    pid[8] = (m > 'L') ? m - 12 : m + 12;
    return GetSymbolByPid(symbol->GetSource(), ProductTypeToString(symbol->GetProductType()), pid);
}

}  // namespace alphaone
