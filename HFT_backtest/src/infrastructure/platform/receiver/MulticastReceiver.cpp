#include "MulticastReceiver.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>

namespace alphaone
{

MulticastReceiver::MulticastReceiver(const char *multicast_ip, uint16_t port,
                                     const char *interface_ip)
    : fd_{socket(AF_INET, SOCK_DGRAM, 0)}, buffer_{'\0'}
{
    if (fd_ < 0)
        throw std::runtime_error("socket create fail");

    u_int yes = 1;
    if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, (char *)&yes, sizeof(yes)) < 0)
        throw std::runtime_error("setsockopt SO_REUSEADDR failed");

    memset(&addr_, 0, sizeof(addr_));
    addr_.sin_family      = AF_INET;
    addr_.sin_addr.s_addr = htonl(INADDR_ANY);
    addr_.sin_port        = htons(port);

    if (bind(fd_, (struct sockaddr *)&addr_, sizeof(addr_)) < 0)
        throw std::runtime_error("bind address failed");
    addr_len_ = sizeof(addr_);

    mreq_.imr_multiaddr.s_addr = inet_addr(multicast_ip);
    mreq_.imr_interface.s_addr = inet_addr(interface_ip);
    if (setsockopt(fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&mreq_, sizeof(mreq_)) < 0)
        throw std::runtime_error("setsockopt IP_ADD_MEMBERSHIP failed");

    read_timeout_.tv_sec  = 0;
    read_timeout_.tv_usec = 10;
    if (setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &read_timeout_, sizeof(read_timeout_)) < 0)
        throw std::runtime_error("setsockopt SO_RCVTIMEO failed");

    if (fcntl(fd_, F_SETFL, O_NONBLOCK) != 0)
        throw std::runtime_error("fail to set non-blocking socket");
}

MulticastReceiver::~MulticastReceiver()
{
    if (fd_ >= 0)
        close(fd_);
}

char *MulticastReceiver::Receive()
{
    int nbytes = recvfrom(fd_, buffer_, 1024, 0, (struct sockaddr *)&addr_, &addr_len_);
    if (nbytes <= 0)
        return nullptr;

    return buffer_;
}


}  // namespace alphaone
