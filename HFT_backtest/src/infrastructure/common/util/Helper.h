#ifndef HELPER_H
#define HELPER_H

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cxxabi.h>
#include <limits>
#include <string.h>
#include <string>

namespace alphaone
{
namespace helper
{
inline std::string demangle(const char *name)
{
    int               status         = -4;
    char *            res            = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    const char *const demangled_name = (status == 0) ? res : name;
    std::string       ret_val(demangled_name);
    free(res);

    ret_val.erase(std::remove(ret_val.begin(), ret_val.end(), ' '), ret_val.end());
    ret_val.erase(std::remove(ret_val.begin(), ret_val.end(), '*'), ret_val.end());

    return ret_val;
}

inline void HexDump(const char *desc, void *addr, int len)
{
    int            i;
    unsigned char  buff[17];
    unsigned char *pc = (unsigned char *)addr;

    // Output description if given.
    if (desc != NULL)
    {
        printf("%s:\n", desc);
    }

    if (len == 0)
    {
        printf("  ZERO LENGTH\n");
        return;
    }

    if (len < 0)
    {
        printf("  NEGATIVE LENGTH: %i\n", len);
        return;
    }

    // Process every byte in the data.
    for (i = 0; i < len; i++)
    {
        // Multiple of 16 means new line (with line offset).
        if ((i % 16) == 0)
        {
            // Just don't print ASCII for the zeroth line.
            if (i != 0)
            {
                printf("  %s\n", buff);
            }

            // Output the offset.
            printf("  %04x ", i);
        }

        // Now the hex code for the specific character.
        printf(" %02x", pc[i]);

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e))
        {
            buff[i % 16] = '.';
        }
        else
        {
            buff[i % 16] = pc[i];
        }
        buff[(i % 16) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % 16) != 0)
    {
        printf("   ");
        i++;
    }

    // And print the final ASCII bit.
    printf("  %s\n", buff);
}

template <typename E>
constexpr auto to_value(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}
}  // namespace helper
}  // namespace alphaone

#endif
