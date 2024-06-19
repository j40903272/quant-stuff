#ifndef STRING_H
#define STRING_H

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace alphaone
{
// split a given string into strings using any character from delimiters string
// if 'compress_tokens'=true then multiple tokens are compresse, i.e.:
//   if compress_tokens=T then "little,brown,,fox" will become {"little","brown","fox}
//   if compress_tokens=F then "little,brown,,fox" will become {"little","brown","","fox"}
void Split(std::vector<std::string> &output, const std::string &input, std::string delimiters,
           bool compress_tokens = false);


std::vector<std::string> SplitIntoVector(const std::string &input, std::string delimiters,
                                         bool compress_tokens);

// split a string into a pair of other strings
// if cannot split cleanly, will exit
std::pair<std::string, std::string> SplitIntoPair(const std::string &input, const char delimiter);

// split a string into a vector of pairs
//   delimiter1 will split into all the vector elements
//   delimited2 will split into the pairs
std::vector<std::pair<std::string, std::string>>
SplitIntoVectorOfPairs(const std::string &input, const char delimiter1, const char delimiter2);

// split a string into an unordered_map
//   delimiter1 will split into all the vector elements
//   delimited2 will split into the pairs
std::unordered_map<std::string, std::string>
SplitIntoUnorderedMap(const std::string &input, const char delimiter1, const char delimiter2);


std::string Concatenate(const std::vector<std::string> &input);

template <typename T>
std::string ConcatenateWithDelimiter(const std::vector<T> &input, const char *delimiter,
                                     bool add_last = false)
{
    std::stringstream ss;
    std::copy(input.begin(), input.end() - 1, std::ostream_iterator<T>(ss, delimiter));
    ss << input.back();
    if (add_last)
        ss << delimiter;
    return ss.str();
}

void Replace(std::string &str, const std::string &from, const std::string &to);

const std::string Replace(const std::string &str, const std::string &from, const std::string &to);

std::vector<std::string_view> SplitSVPtr(std::string_view &str, std::string_view delims);

void Remove(std::string &str, const char delimiter);

std::string StringAligningCenter(const std::string &target, const size_t size);
}  // namespace alphaone

#endif
