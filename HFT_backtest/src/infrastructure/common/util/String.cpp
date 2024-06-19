#include "String.h"

#include <algorithm>  // std::find_if
#include <boost/algorithm/string.hpp>
#include <numeric>
#include <string>
#include <vector>

namespace alphaone
{
void Split(std::vector<std::string> &output, const std::string &input, std::string delimiters,
           bool compress_tokens)
{
    if (input == "")
    {
        return;
    }

    size_t start{0};
    size_t end{0};
    while (end != std::string::npos && start != std::string::npos)
    {
        end = input.find_first_of(delimiters, start);
        output.push_back(
            input.substr(start, (end == std::string::npos) ? std::string::npos : end - start));
        start = (compress_tokens ? input.find_first_not_of(delimiters, end) : end + 1);
    }
}

std::vector<std::string> SplitIntoVector(const std::string &input, std::string delimiters,
                                         bool compress_tokens)
{
    std::vector<std::string> output;
    Split(output, input, delimiters, compress_tokens);
    return output;
}

std::pair<std::string, std::string> SplitIntoPair(const std::string &input, const char delimiter)
{
    const size_t i{input.find_first_of(delimiter)};
    return std::make_pair(input.substr(0, i), (i == std::string::npos) ? "" : input.substr(i + 1));
}

std::vector<std::pair<std::string, std::string>>
SplitIntoVectorOfPairs(const std::string &input, const char delimiter1, const char delimiter2)
{
    std::vector<std::string> temp_vector;
    Split(temp_vector, input, std::string(1, delimiter1));
    std::vector<std::pair<std::string, std::string>> return_vector;
    for (const auto &str : temp_vector)
    {
        return_vector.push_back(SplitIntoPair(str, delimiter2));
    }
    return return_vector;
}

std::unordered_map<std::string, std::string>
SplitIntoUnorderedMap(const std::string &input, const char delimiter1, const char delimiter2)
{
    std::vector<std::string> temp_vector;
    Split(temp_vector, input, std::string(1, delimiter1));
    std::unordered_map<std::string, std::string> return_unordered_map;
    for (const auto &str : temp_vector)
    {
        return_unordered_map.insert(SplitIntoPair(str, delimiter2));
    }
    return return_unordered_map;
}

std::string Concatenate(const std::vector<std::string> &input)
{
    return std::accumulate(input.begin(), input.end(), std::string{""},
                           [](std::string s, const std::string &in) -> decltype(auto)
                           { return s += in; });
}

void Replace(std::string &str, const std::string &from, const std::string &to)
{
    size_t ps{0};
    while ((ps = str.find(from, ps)) != std::string::npos)
    {
        str.replace(ps, from.length(), to);
        ps += to.length();  // handle cases where 'to' is a substring of 'from'
    }
    return;
}

const std::string Replace(const std::string &str, const std::string &from, const std::string &to)
{
    std::string tmp{str};
    size_t      ps{0};
    while ((ps = tmp.find(from, ps)) != std::string::npos)
    {
        tmp.replace(ps, from.length(), to);
        ps += to.length();  // handle cases where 'to' is a substring of 'from'
    }
    return tmp;
}

std::vector<std::string_view> SplitSVPtr(std::string_view &str, std::string_view delims)
{
    std::vector<std::string_view> output;

    for (auto first = str.data(), second = str.data(), last = first + str.size();
         second != last && first != last; first = second + 1)
    {
        second = std::find_first_of(first, last, std::cbegin(delims), std::cend(delims));

        if (first != second)
            output.emplace_back(first, second - first);
    }

    return output;
}

void Remove(std::string &str, const char delimiter)
{
    str.erase(std::remove(str.begin(), str.end(), delimiter), str.end());
}

std::string StringAligningCenter(const std::string &target, const size_t size)
{
    std::string result{target};

    if (const int diff{static_cast<int>(size - target.size())}; diff > 0)
    {
        const size_t head{static_cast<size_t>(diff / 2)};
        const size_t tail{static_cast<size_t>(diff - head)};
        for (size_t i{0}; i < head; ++i)
        {
            result = " " + result;
        }
        for (size_t i{0}; i < tail; ++i)
        {
            result = result + " ";
        }
    }

    return result;
}
}  // namespace alphaone
