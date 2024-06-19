#ifndef _MULTICASTSENDER_H_
#define _MULTICASTSENDER_H_

#include <arpa/inet.h>
#include <sys/socket.h>

namespace alphaone
{

class MulticastSender
{

  public:
    MulticastSender(const char *ip, uint16_t port, const char *interface_ip);
    ~MulticastSender();
    ssize_t Send(const void *buff, size_t len);

  private:
    int                fd_;
    struct sockaddr_in addr_;
    struct in_addr     imr_;
};

}  // namespace alphaone


#endif