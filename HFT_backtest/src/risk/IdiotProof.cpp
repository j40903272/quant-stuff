#include "IdiotProof.h"

namespace alphaone
{
IdiotProof::IdiotProof(const std::string &sys_id, const void *source, const std::string &server,
                       Strategy *strategy)
    : CommandSource{server, "AlphaOneIdiotProof", 1.0}
    , id_{sys_id}
    , source_{source}
    , strategy_{strategy}
{
    risk_parameters_["sysid"]  = id_;
    risk_parameters_["source"] = reinterpret_cast<size_t>(source);

    if (strategy != nullptr)
    {
        strategy_->AddIdiotProof(this);
    }
}

IdiotProof::IdiotProof(const std::string &sys_id, const void *source,
                       std::shared_ptr<NATSConnection> conn, Strategy *strategy)
    : CommandSource{conn, "AlphaOneIdiotProof", 1.0}
    , id_{sys_id}
    , source_{source}
    , strategy_{strategy}
{
    risk_parameters_["sysid"]  = id_;
    risk_parameters_["source"] = reinterpret_cast<size_t>(source);

    if (strategy != nullptr)
    {
        strategy_->AddIdiotProof(this);
    }
}

IdiotProof::~IdiotProof()
{
}

void IdiotProof::Dump()
{
#ifdef DebugIdiotProof
    // print
    std::cout << risk_parameters_.dump(4, ' ') << std::endl;
#endif

    // write
    const std::string fileroot{RISK_MANAGEMENT_FILE_ROOT};
    const std::string filename{
        (strategy_ != nullptr ? strategy_->GetEngine()->GetDate().to_string() + "." : "") + id_ +
        "." + std::to_string(reinterpret_cast<size_t>(source_)) + ".json"};
    if (CheckDirectory(fileroot))
    {
        DumpToFile(fileroot + "/" + filename, risk_parameters_.dump(4, ' '));
    }

    // publish
    CommandSource::Publish(risk_parameters_);
}
}  // namespace alphaone
