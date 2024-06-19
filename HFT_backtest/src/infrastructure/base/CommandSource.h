#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/nats/NATSConnection.h"

namespace alphaone
{
class CommandSource
{
  public:
    CommandSource(const std::string &server, const std::string &object, const double delay);
    CommandSource(std::shared_ptr<NATSConnection> conn, const std::string &object,
                  const double delay);
    CommandSource(const CommandSource &) = delete;
    CommandSource &operator=(const CommandSource &) = delete;

    virtual ~CommandSource();

    virtual void Publish(const std::string &word);
    virtual void Publish(const nlohmann::json &js);

  protected:
    void PublishSingleJson(const nlohmann::json &js);

    const std::string               server_;
    const std::string               object_;
    const double                    delay_;
    std::shared_ptr<NATSConnection> connection_;
    int                             maximum_retry_;
};
}  // namespace alphaone
