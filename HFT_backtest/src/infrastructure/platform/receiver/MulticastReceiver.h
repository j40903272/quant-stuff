#ifndef _MULTICASTRECEIVER_H_
#define _MULTICASTRECEIVER_H_

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace alphaone
{

class MulticastReceiver
{
  public:
    MulticastReceiver(const char *multicast_ip, uint16_t port, const char *interface_ip);
    ~MulticastReceiver();
    char *Receive();

  private:
    int                fd_;
    struct sockaddr_in addr_;
    socklen_t          addr_len_;
    struct ip_mreq     mreq_;
    struct timeval     read_timeout_;
    char               buffer_[1024];
};


}  // namespace alphaone


#endif