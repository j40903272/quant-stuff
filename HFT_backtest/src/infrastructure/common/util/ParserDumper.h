#ifndef _PARSERDUMPER_H_
#define _PARSERDUMPER_H_

#include "infrastructure/common/json/Json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace alphaone
{
inline nlohmann::json ReadJson(const std::filesystem::path &path)
{
    nlohmann::json json;
    {
        std::ifstream fs{path};
        if (!fs.is_open() || fs.fail())
        {
            std::cout << "failed to open file=" << path << std::endl;
            abort();
        }
        fs >> json;
        fs.close();
    }
    return json;
}

inline void DumpToFile(const std::string &path, const std::string &contents)
{
    std::ofstream fs{path};
    if (!fs.is_open() || fs.fail())
    {
        std::cout << "failed to open file=" << path << std::endl;
        abort();
    }

    fs << contents << std::endl;
    fs.close();
}

inline void DumpToFile(const std::string &path, const std::stringstream &contents)
{
    DumpToFile(path, contents.str());
}

inline bool CheckDirectory(const std::string &path)
{
    try
    {
        if (!std::filesystem::exists(path))
        {
            std::filesystem::create_directories(path);
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}
}  // namespace alphaone

#endif
