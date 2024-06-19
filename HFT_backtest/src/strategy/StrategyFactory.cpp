#include "StrategyFactory.h"

namespace alphaone
{
StrategyFactory::StrategyFactory(ObjectManager *      object_manager,
                                 MultiBookManager *   multi_book_manager,
                                 MultiCounterManager *multi_counter_manager, Engine *engine,
                                 std::vector<TaifexOrderManagerBase *> *taifex_order_managers)
    : object_manager_{object_manager}
    , multi_book_manager_{multi_book_manager}
    , multi_counter_manager_{multi_counter_manager}
    , engine_{engine}
    , taifex_order_managers_{taifex_order_managers}
{
}

StrategyFactory::~StrategyFactory()
{
    for (auto &strategy : strategies_)
    {
        if (strategy)
        {
            delete strategy;
            strategy = nullptr;
        }
    }
}

Strategy *StrategyFactory::CreateStrategy(const std::string &name, size_t config_index)
{
    if (name == "CrossExchangeArbitrageStrategy")
    {
        strategies_.push_back(new CrossExchangeArbitrageStrategy(
            object_manager_, multi_book_manager_, multi_counter_manager_, engine_,
            (*taifex_order_managers_)[config_index]));
        return strategies_.back();
    }

    if (name == "TestMakerStrategy")
    {
        strategies_.push_back(new TestMakerStrategy(
            object_manager_, multi_book_manager_, multi_counter_manager_, engine_, 
            (*taifex_order_managers_)[config_index], config_index));
        return strategies_.back();
    }

    throw std::invalid_argument("unrecognized strategy " + name);
}
}  // namespace alphaone
