#ifndef _BOOKDATASOURCE_H_
#define _BOOKDATASOURCE_H_

#include "infrastructure/common/typedef/Typedefs.h"

namespace alphaone
{
class BookDataListener;

class BookDataSource
{
  public:
    virtual void AddPreBookListener(BookDataListener *listener)  = 0;
    virtual void AddPostBookListener(BookDataListener *listener) = 0;

    // get book type -- need to be defined explicitly;
    virtual DataSourceType GetType() const = 0;
};
}  // namespace alphaone

#endif
