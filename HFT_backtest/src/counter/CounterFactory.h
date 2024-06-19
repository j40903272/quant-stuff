#ifndef _COUNTERFACTORY_H_
#define _COUNTERFACTORY_H_

#include "infrastructure/platform/counter/Counter.h"
#include "infrastructure/platform/manager/MultiBookManager.h"
#include "infrastructure/platform/manager/ObjectManager.h"

namespace alphaone
{
class CounterFactory
{
  public:
    ~CounterFactory() = default;

    static Counter *RetrieveCounterFromCounterSpec(const ObjectManager *object_manager,
                                                   MultiBookManager *   multi_book_manager,
                                                   Engine *engine, const std::string &interval,
                                                   const nlohmann::json &spec);

  protected:
    CounterFactory()                       = default;
    CounterFactory(const CounterFactory &) = delete;
    CounterFactory &operator=(const CounterFactory &) = delete;
    CounterFactory(CounterFactory &&)                 = delete;
    CounterFactory &operator=(CounterFactory &&) = delete;
};
}  // namespace alphaone

#endif
