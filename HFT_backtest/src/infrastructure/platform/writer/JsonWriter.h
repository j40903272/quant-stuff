#ifndef _JSONWRITER_H_
#define _JSONWRITER_H_

#include "infrastructure/common/datetime/Date.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/util/Branch.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace alphaone
{

enum ReportType
{
    Event          = 0,
    PnL            = 1,
    LastReportType = 2,
};

class JsonWriter
{

  public:
    JsonWriter(const std::filesystem::path &file_path, size_t cache_size = 64);
    JsonWriter(const std::filesystem::path &file_path, const std::string &name,
               size_t cache_size = 64);
    JsonWriter() = delete;
    ~JsonWriter();
    void Init(const std::filesystem::path &file_path);
    void InsertReport(const nlohmann::json &json);
    void InsertByKey(const std::string &key, const nlohmann::json &json);
    void InsertByKey(const std::string &key, const std::string &value);
    void SetName(const std::string &name);
    void Flush();

  private:
    std::string                                  name_;
    std::fstream                                 file_[LastReportType];
    int                                          count_;
    std::map<std::string, std::string>           key_to_json_;
    std::vector<std::shared_ptr<nlohmann::json>> cache_;
    size_t                                       cache_count_;
};

}  // namespace alphaone

#endif
