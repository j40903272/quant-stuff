#ifndef _IDIOTPROOF_H_
#define _IDIOTPROOF_H_

#include "infrastructure/base/CommandSource.h"
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/util/ParserDumper.h"
#include "strategy/Strategy.h"

// #define DebugIdiotProof

namespace alphaone
{
class Strategy;

class IdiotProof : public CommandSource
{
  public:
    using CommandSource::Publish;

    IdiotProof(const std::string &sys_id, const void *source, const std::string &server,
               Strategy *strategy = nullptr);
    IdiotProof(const std::string &sys_id, const void *source, std::shared_ptr<NATSConnection> conn,
               Strategy *strategy = nullptr);
    IdiotProof(const IdiotProof &) = delete;
    IdiotProof &operator=(const IdiotProof &) = delete;
    IdiotProof(IdiotProof &&)                 = delete;
    IdiotProof &operator=(IdiotProof &&) = delete;

    ~IdiotProof();

    template <typename type>
    void Update(const std::string &key, const type &value);
    void Update(const nlohmann::json &json);
    void Dump();

  private:
    const std::string RISK_MANAGEMENT_FILE_ROOT{std::string("/var/files/risk/")};

    const std::string id_;
    const void *      source_;

    Strategy *strategy_;

    nlohmann::json risk_parameters_;
};

template <typename type>
void IdiotProof::Update(const std::string &key, const type &value)
{
    risk_parameters_[key] = value;
    Dump();
}

inline void IdiotProof::Update(const nlohmann::json &json)
{
    risk_parameters_.update(json);
    Dump();
}
}  // namespace alphaone

#endif
