#include "MulticastSender.h"

#include <cstring>
#include <stdexcept>
#include <unistd.h>

namespace alphaone
{

MulticastSender::MulticastSender(const char *ip, uint16_t port, const char *interface_ip)
{
    // set up multicast
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0)
        throw std::runtime_error("socket create fail");

    memset(&addr_, 0, sizeof(addr_));
    addr_.sin_family      = AF_INET;
    addr_.sin_addr.s_addr = inet_addr(ip);
    addr_.sin_port        = htons(port);

    // set up multicast interface
    struct in_addr imr;
    imr.s_addr = inet_addr(interface_ip);
    if (setsockopt(fd_, IPPROTO_IP, IP_MULTICAST_IF, &imr.s_addr, sizeof(struct in_addr)) < 0)
    {
        auto msg = std::string("Set Multicast Interface failed");
        perror(msg.c_str());
        throw std::runtime_error(msg);
    }
}

MulticastSender::~MulticastSender()
{
    if (fd_ >= 0)
        close(fd_);
}

ssize_t MulticastSender::Send(const void *buff, size_t len)
{
    return sendto(fd_, buff, len, 0, (struct sockaddr *)&addr_, sizeof(addr_));
}

}  // namespace alphaone
