#ifndef _STRATEGYFACTORY_H_
#define _STRATEGYFACTORY_H_

#include "strategy/Strategy.h"
#include "strategy/GenericFactory.h"
#include "strategy/CrossExchangeArbitrageStrategy/CrossExchangeArbitrageStrategy.h"
// #include "strategy/TestHedger/TestHedger.h"
#include "strategy/TestMakerStrategy/TestMakerStrategy.h"
// #include "strategy/TestStrategy/TestStrategy.h"
// #include "strategy/TickToSignal/TickToSignal.h"
// #include "strategy/TickToTrade/TickToTrade.h"

namespace alphaone
{

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)

#define REGISTER_STRATEGY(ST)                                                                      \
    static const bool CONCAT(strategy, __COUNTER__) =                                              \
        alphaone::GenericFactory<                                                                  \
            alphaone::Strategy, alphaone::ObjectManager *, alphaone::MultiBookManager *,           \
            alphaone::MultiCounterManager *, alphaone::Engine *,            \
            alphaone::TaifexOrderManagerBase *>::Instance()          \
            .Register(#ST,                                                                         \
                      [](alphaone::ObjectManager *      object_manager,                            \
                         alphaone::MultiBookManager *   multi_book_manager,                        \
                         alphaone::MultiCounterManager *multi_counter_manager,                     \
                         alphaone::Engine *engine,                \
                         alphaone::TaifexOrderManagerBase *order_manager) -> ST * {                 \
                          return new ST(object_manager, multi_book_manager, multi_counter_manager, \
                                        engine, order_manager);    \
                      });

inline alphaone::Strategy *CreateStrategy(const std::string &            type,
                                          alphaone::ObjectManager *      object_manager,
                                          alphaone::MultiBookManager *   multi_book_manager,
                                          alphaone::MultiCounterManager *multi_counter_manager,
                                          alphaone::Engine *engine,
                                          alphaone::TaifexOrderManagerBase *order_manager)
{
    return alphaone::GenericFactory<
               alphaone::Strategy, alphaone::ObjectManager *, alphaone::MultiBookManager *,
               alphaone::MultiCounterManager *, alphaone::Engine *,
               alphaone::TaifexOrderManagerBase *>::Instance()
        .Create(type, object_manager, multi_book_manager, multi_counter_manager, engine,
                order_manager);
}

class StrategyFactory
{
  public:
    StrategyFactory(ObjectManager *object_manager, MultiBookManager *multi_book_manager,
                    MultiCounterManager *multi_counter_manager, Engine *engine,
                    std::vector<TaifexOrderManagerBase *> *taifex_order_managers);
    StrategyFactory(const StrategyFactory &strategy_factory) = delete;
    StrategyFactory &operator=(const StrategyFactory &strategy_factory) = delete;

    ~StrategyFactory();

    // get new strategy pointer from factory
    Strategy *CreateStrategy(const std::string &name, size_t config_index = 0);

  private:
    ObjectManager *                        object_manager_;
    MultiBookManager *                     multi_book_manager_;
    MultiCounterManager *                  multi_counter_manager_;
    Engine *                               engine_;
    std::vector<TaifexOrderManagerBase *> *taifex_order_managers_;
    std::vector<Strategy *>                strategies_;
};

}  // namespace alphaone

#endif
