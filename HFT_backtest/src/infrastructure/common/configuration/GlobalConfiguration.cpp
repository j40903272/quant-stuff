#include "GlobalConfiguration.h"

#include "infrastructure/common/symbol/Symbol.h"
#include "infrastructure/common/util/Logger.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>


namespace alphaone
{
GlobalConfiguration::GlobalConfiguration()
{
}

GlobalConfiguration::~GlobalConfiguration()
{
}

void GlobalConfiguration::Load(const char *file_root_path)
{
    Load(std::string(file_root_path));
}

void GlobalConfiguration::Load(const std::string &file_path)
{
    file_root_path_ = file_path;

    if (!std::filesystem::exists(file_root_path_))
    {
        throw std::invalid_argument("Trying to read global configuration but not exist at " +
                                    file_root_path_);
    }

    std::ifstream fs{file_root_path_};
    if (fs.is_open() && !fs.fail())
    {
        fs >> json_;
        fs.close();
    }
    else
    {
        throw std::invalid_argument("file stream is not open for file " + file_root_path_);
    }
}

void GlobalConfiguration::Load(const nlohmann::json &json)
{
    json_ = json;
}

void GlobalConfiguration::Load(const GlobalConfiguration *json)
{
    json_ = json->GetJson();
}
}  // namespace alphaone
