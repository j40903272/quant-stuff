#ifndef _SYMBOLMANAGER_H
#define _SYMBOLMANAGER_H

#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/symbol/Symbol.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace alphaone
{
class SymbolManager
{
  public:
    SymbolManager();
    SymbolManager(const SymbolManager &) = delete;
    SymbolManager &operator=(const SymbolManager &) = delete;
    SymbolManager(SymbolManager &&)                 = delete;
    SymbolManager &operator=(SymbolManager &&) = delete;

    ~SymbolManager();

    void Load(const std::filesystem::path &file_root_path);

    const Symbol *GetSymbolByPid(const std::string &source, const std::string &type,
                                 const std::string &pid) const;
    const Symbol *GetSymbolByCPid(const std::string &source, const std::string &type,
                                  const std::string &cpid) const;
    const Symbol *GetSymbolByString(const std::string &symbol) const;

    std::vector<const Symbol *> GetSymbolsBySource(const std::string &source,
                                                   const std::string &type) const;
    std::vector<const Symbol *> GetSymbolsBySourceAndGroup(const std::string &source,
                                                           const std::string &type,
                                                           const std::string &group) const;
    const std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::unordered_map<std::string, const Symbol *>>> &
                                 GetPidMap() const;
    const std::filesystem::path &GetJsonPath() const;
    const Date &                 GetDate() const;
    const Symbol *               GetOptionPairSymbol(const Symbol *symbol) const;

  private:
    std::filesystem::path file_path_;
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::unordered_map<std::string, const Symbol *>>>
        pid_to_symbol_map_;

    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::unordered_map<std::string, const Symbol *>>>
        cpid_to_symbol_map_;

    std::vector<const Symbol *> created_symbols_;
    Date                        date_;

    void CreateTaifexRollSymbol(const nlohmann::json &config);
    void CreateTPriceSymbol(const nlohmann::json &config);
    void CreateTWSEDataFileSymbol(const nlohmann::json &config);
};
}  // namespace alphaone
#endif
