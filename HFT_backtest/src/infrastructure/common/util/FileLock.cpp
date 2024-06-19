#include "FileLock.h"

namespace alphaone
{

FileLock::FileLock(const std::string &system_id, int index, std::filesystem::path root_path)
    : system_id_{system_id}, index_{index}, root_path_{root_path}
{
    std::filesystem::path lock_file_path{
        root_path / (system_id_ + std::string("_") + std::to_string(index) + ".lock")};
    int fd = open(lock_file_path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0)
    {
        throw std::runtime_error("Failed to open lock file!");
    }

    struct flock fl;
    fl.l_start  = 0;
    fl.l_len    = 0;
    fl.l_type   = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_pid    = getpid();
    if (fcntl(fd, F_SETLK, &fl) < 0)
    {
        throw std::runtime_error("This AP is already running!");
    }
}

}  // namespace alphaone
