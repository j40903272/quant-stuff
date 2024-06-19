#ifndef _NET_H_
#define _NET_H_
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

std::vector<std::string> GetAllIP();
#endif