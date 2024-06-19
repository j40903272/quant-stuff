#ifndef _GENERICFACTORY_H_
#define _GENERICFACTORY_H_

#include "infrastructure/common/util/Logger.h"

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

namespace alphaone
{
template <class Object, class... Args>
class GenericFactory
{
  public:
    ~GenericFactory() = default;

    static GenericFactory &Instance()
    {
        static GenericFactory factory;
        return factory;
    }

    bool Register(const std::string &key, std::function<Object *(Args... args)> function)
    {
        spdlog::set_pattern("[%^%l%$] %v");
        function_map_[key] = function;
        SPDLOG_DEBUG("register {} into function map", key);

        return true;
    }

    Object *Create(const std::string &key, Args... args)
    {
        SPDLOG_DEBUG("create {} from function map", key);

        if (function_map_.find(key) == function_map_.end())
        {
            return nullptr;
        }

        return function_map_.at(key)(std::forward<Args>(args)...);
    }

  private:
    std::unordered_map<std::string, std::function<Object *(Args... args)>> function_map_;
    GenericFactory() = default;
};
}  // namespace alphaone
#endif
