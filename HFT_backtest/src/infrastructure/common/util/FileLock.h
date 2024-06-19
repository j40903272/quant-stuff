#ifndef _FILELOCK_H_
#define _FILELOCK_H_

#include <fcntl.h> /* fcntl, open */
#include <filesystem>
#include <string>
#include <unistd.h> /* getpid */

namespace alphaone
{

class FileLock
{
  public:
    FileLock() = delete;
    FileLock(const std::string &system_id, int index, std::filesystem::path root_path = "./");
    ~FileLock() = default;

  private:
    /* data */
    const std::string           system_id_;
    const int                   index_;
    const std::filesystem::path root_path_;
};

}  // namespace alphaone

#endif