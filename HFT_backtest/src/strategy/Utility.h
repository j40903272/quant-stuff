#ifndef _STRATEGY_UTILITY_H_
#define _STRATEGY_UTILITY_H_

#include <string>

namespace alphaone
{

inline std::string PadUserdefine(std::string userdefine)
{
    userdefine.resize(8, ' ');
    return userdefine;
}

#endif


}  // namespace alphaone
