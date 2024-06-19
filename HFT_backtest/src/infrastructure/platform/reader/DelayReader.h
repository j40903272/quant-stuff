#ifndef _DELAYREADER_H_
#define _DELAYREADER_H_

#include "infrastructure/common/file/DelimitedFileReader.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/spdlog/spdlog.h"

#include <map>
#include <unordered_map>

namespace alphaone
{

enum class SeqNoMode
{
    Exact      = 0,
    LowerBound = 1,
    UpperBound = 2,
    LastMode   = 3
};

enum class CalcMode
{
    LowerBound = 0,
    UpperBound = 1,
    LastMode   = 2
};

class DelayReader
{

  public:
    DelayReader(const nlohmann::json &config);
    ~DelayReader();

    double GetDelay(int seqno, long double calc_time);
    void   Dump() const;

  private:
    static const inline std::unordered_map<std::string, SeqNoMode> seq_mode_map_{
        {"Exact", SeqNoMode::Exact},
        {"LowerBound", SeqNoMode::LowerBound},
        {"UpperBound", SeqNoMode::UpperBound}};
    static const inline std::unordered_map<std::string, CalcMode> calc_mode_map_{
        {"LowerBound", CalcMode::LowerBound}, {"UpperBound", CalcMode::UpperBound}};

    DelimitedFileReader reader_;

    std::map<int64_t, std::map<long double, long double>> seqno_to_delay_map_;

    const double default_delay_;
    SeqNoMode    seq_mode_;
    CalcMode     calc_mode_;
};


}  // namespace alphaone


#endif