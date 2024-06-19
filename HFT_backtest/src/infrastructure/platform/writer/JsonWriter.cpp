#include "JsonWriter.h"

namespace alphaone
{

JsonWriter::JsonWriter(const std::filesystem::path &file_path, size_t cache_size)
    : name_{"writer"}, count_{0}, cache_count_{0}
{
    Init(file_path);
    cache_.resize(cache_size);
}

JsonWriter::JsonWriter(const std::filesystem::path &file_path, const std::string &name,
                       size_t cache_size)
    : name_{name}, count_{0}, cache_count_{0}
{
    Init(file_path);
    cache_.resize(cache_size);
}

JsonWriter::~JsonWriter()
{
    if (cache_count_)
    {
        Flush();
    }
    file_[Event] << "]";
    file_[PnL] << "{";
    for (auto it = key_to_json_.begin(); it != key_to_json_.end(); ++it)
    {
        file_[PnL] << it->first << ":" << it->second << ",\n";
    }
    file_[PnL] << "\"name\":\"" << name_ << "\"}";
    for (int t = Event; t < LastReportType; ++t)
    {
        file_[t].close();
    }
}

void JsonWriter::Init(const std::filesystem::path &file_path)
{
    const std::filesystem::path path[LastReportType] = {file_path.string() + ".event",
                                                        file_path.string() + ".pnl"};
    if (file_path.has_filename())
    {
        for (int t = Event; t < LastReportType; ++t)
        {
            file_[t].open(path[t], std::ios::out);
        }
    }
    else
    {
        const std::filesystem::path new_path[LastReportType]{
            file_path / (name_ + ".event.report.json"), file_path / (name_ + ".pnl.report.json")};
        for (int t = Event; t < LastReportType; ++t)
        {
            file_[t].open(new_path[t], std::ios::out);
        }
    }

    for (int t = Event; t < LastReportType; ++t)
    {
        if (!file_[t].is_open() || file_[t].fail())
        {
            SPDLOG_ERROR("Failed to open file at path {}", path[t]);
            abort();
        }
    }
    // use a big json array to cover all write in json
    file_[Event] << "[";
}

void JsonWriter::InsertReport(const nlohmann::json &json)
{
    cache_[cache_count_++] = std::make_shared<nlohmann::json>(json);
    if (BRANCH_UNLIKELY(cache_count_ >= cache_.size()))
    {
        Flush();
    }
}

void JsonWriter::InsertByKey(const std::string &key, const nlohmann::json &json)
{
    if (auto it = key_to_json_.insert({"\"" + key + "\"", json.dump()}); !it.second)
    {
        SPDLOG_WARN("Insert key {} is duplicate, going to overwrite previous data", key);
        it.first->second = std::move(json.dump());
    }
}

void JsonWriter::InsertByKey(const std::string &key, const std::string &value)
{
    if (auto it = key_to_json_.insert({"\"" + key + "\"", "\"" + value + "\""}); !it.second)
    {
        SPDLOG_WARN("Insert key {} is duplicate, going to overwrite previous data");
        it.first->second = std::move(value);
    }
}

void JsonWriter::SetName(const std::string &name)
{
    name_ = name;
}

void JsonWriter::Flush()
{
    std::stringstream ss;
    for (size_t c{0}; c < cache_count_; ++c)
    {
        ss << (count_++ ? ",\n" : "") << *(cache_[c]);
    }
    file_[Event] << ss.str();
    cache_count_ = 0;
}

}  // namespace alphaone
