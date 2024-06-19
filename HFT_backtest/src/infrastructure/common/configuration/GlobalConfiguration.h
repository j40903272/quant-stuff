#ifndef _GLOBALCONFIG_H_
#define _GLOBALCONFIG_H_

#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/typedef/Typedefs.h"

#include <string>

namespace alphaone
{
class GlobalConfiguration
{
  public:
    GlobalConfiguration();
    ~GlobalConfiguration();
    void Load(const char *file_root_path);
    void Load(const std::string &file_root_path);
    void Load(const nlohmann::json &json);
    void Load(const GlobalConfiguration *json);

    const std::string GetPath() const
    {
        return file_root_path_;
    }

    const nlohmann::json &GetJson() const
    {
        return json_;
    }

    template <typename Value>
    void Overwrite(const std::string &key, const Value &value)
    {
        if (json_.contains(key))
        {
            auto v = nlohmann::json(value);
            SPDLOG_WARN("[{}] {} from {} to {}", __func__, key, json_[key].dump(), v.dump());
            json_[key] = v;
        }
        else
        {
            SPDLOG_ERROR("[{}] Cannot find key {} in json", __func__, key);
        }
    }

  protected:
    std::string    file_root_path_;
    nlohmann::json json_;
};
}  // namespace alphaone
#endif
