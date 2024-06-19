#include "DelayReader.h"

namespace alphaone
{

DelayReader::DelayReader(const nlohmann::json &config)
    : reader_{config.value("file_path", "delay.csv")}
    , default_delay_{config.value("default_delay", 8e-4)}
    , seq_mode_{SeqNoMode::Exact}
    , calc_mode_{CalcMode::LowerBound}
{
    // parse mode
    if (config.contains("seq_mode"))
    {
        const auto &seq = config["seq_mode"];
        if (seq.is_string())
        {
            auto sit = seq_mode_map_.find(seq.get<std::string>());
            if (sit != seq_mode_map_.end())
                seq_mode_ = sit->second;
            else
                throw std::invalid_argument(fmt::format("Unsupported seq mode {}", seq));
        }
        else if (seq.is_number_integer())
        {
            auto smode = seq.get<int>();
            if (smode >= static_cast<int>(SeqNoMode::Exact) &&
                smode < static_cast<int>(SeqNoMode::LastMode))
            {
                seq_mode_ = static_cast<SeqNoMode>(smode);
            }
            else
            {
                throw std::invalid_argument(fmt::format("Unsupported seq mode {}", seq));
            }
        }
        else
        {
            throw std::invalid_argument(fmt::format("Unsupported seq mode {}", seq));
        }
    }

    if (config.contains("calc_mode"))
    {
        const auto &calc = config["calc_mode"];
        if (calc.is_string())
        {
            auto cit = calc_mode_map_.find(calc.get<std::string>());
            if (cit != calc_mode_map_.end())
                calc_mode_ = cit->second;
            else
                throw std::invalid_argument(fmt::format("Unsupported calc mode {}", calc));
        }
        else if (calc.is_number_integer())
        {
            auto cmode = calc.get<int>();
            if (cmode >= static_cast<int>(SeqNoMode::Exact) &&
                cmode < static_cast<int>(SeqNoMode::LastMode))
            {
                seq_mode_ = static_cast<SeqNoMode>(cmode);
            }
            else
            {
                throw std::invalid_argument(fmt::format("Unsupported calc mode {}", calc));
            }
        }
        else
        {
            throw std::invalid_argument(fmt::format("Unsupported calc mode {}", calc));
        }
    }

    while (reader_.ReadNext())
    {
        const auto &entries = reader_.GetSplitEntries();
        auto        calc    = std::stold(entries[3]) - std::stold(entries[2]);
        auto        delay   = std::stold(entries[4]) - std::stold(entries[2]);
        if (calc < 0. || delay < 0.)
            continue;
        auto [sit, is_success] = seqno_to_delay_map_.insert(
            {std::stol(entries[0]), std::map<long double, long double>{{calc, delay}}});
        if (!is_success)
            sit->second.emplace(calc, delay);
    }
}

DelayReader::~DelayReader()
{
}

double DelayReader::GetDelay(int seqno, long double calc_time)
{
    auto [lower, upper] = seqno_to_delay_map_.equal_range(seqno);
    std::map<long double, long double> *cmap{nullptr};
    if (seq_mode_ == SeqNoMode::Exact && lower->first == seqno)
        cmap = &lower->second;
    else if (seq_mode_ == SeqNoMode::LowerBound && lower != seqno_to_delay_map_.end())
        cmap = &lower->second;
    else if (seq_mode_ == SeqNoMode::UpperBound && upper != seqno_to_delay_map_.end())
        cmap = &upper->second;

    if (!cmap)
        return default_delay_;

    auto [clower, cupper] = cmap->equal_range(calc_time);
    if (calc_mode_ == CalcMode::LowerBound && clower != cmap->end())
        return clower->second;
    if (calc_mode_ == CalcMode::UpperBound && cupper != cmap->end())
        return cupper->second;

    return default_delay_;
}

void DelayReader::Dump() const
{
    SPDLOG_INFO("SeqNo,CalcTime,DelayTime");
    for (const auto &[seq, calc_to_delay] : seqno_to_delay_map_)
    {
        for (const auto &[calc, delay] : calc_to_delay)
        {
            SPDLOG_INFO("{},{:.9f},{:.9f}", seq, calc, delay);
        }
    }
}

}  // namespace alphaone
