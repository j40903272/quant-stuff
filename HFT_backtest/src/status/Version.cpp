#include "Version.h"

namespace alphaone
{
namespace Version
{
std::string GetVersionInfo()
{
    std::ostringstream buffer;
    buffer.str("");

    buffer << "[Version] branch=" << GIT_BRANCH << " last_commit_date=" << GIT_COMMIT_DATE
           << " last_commit_hash=" << GIT_COMMIT_HASH;

    return buffer.str();
}
}  // namespace Version
}  // namespace alphaone
