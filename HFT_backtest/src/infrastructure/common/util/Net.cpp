#include "infrastructure/common/util/Net.h"

std::vector<std::string> GetAllIP()
{
    std::vector<std::string> ret;
    struct ifaddrs *         ifAddrStruct = NULL;
    struct ifaddrs *         ifa          = NULL;
    void *                   tmpAddrPtr   = NULL;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr)
        {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET)
        {
            tmpAddrPtr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            ret.push_back(addressBuffer);
        }
    }
    if (ifAddrStruct != NULL)
    {
        freeifaddrs(ifAddrStruct);
    }
    return ret;
}