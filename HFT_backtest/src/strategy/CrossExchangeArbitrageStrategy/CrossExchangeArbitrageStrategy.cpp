#include "CrossExchangeArbitrageStrategy.h"

#include "infrastructure/common/math/Constant.h"
#include "infrastructure/common/math/Math.h"
#include "infrastructure/platform/book/MarketByOrderBook.h"

// #define TRACKING_STRATEGY_MODEL

namespace alphaone
{
CrossExchangeArbitrageStrategy::CrossExchangeArbitrageStrategy(ObjectManager *object_manager, MultiBookManager *multi_book_manager,
                           MultiCounterManager *multi_counter_manager, Engine *engine, TaifexOrderManagerBase *order_manager)
    : Strategy{"CrossExchangeArbitrageStrategy", object_manager, multi_book_manager, multi_counter_manager,
               engine,     order_manager}
{

    const nlohmann::json json(
        object_manager->GetGlobalConfiguration()->GetJson().at("Strategy"));
    std::cout << "gogogogogogogo" << std::endl;
}

void CrossExchangeArbitrageStrategy::OnPacketEnd(const Timestamp                 event_loop_time,
                                  const BookDataMessagePacketEnd *o)
{
    // SPDLOG_INFO("strategy OnPacketEnd symbol {}", o->GetSymbol());
    std::cout << "wtf:" << o->GetSymbol() << std::endl;
}

CrossExchangeArbitrageStrategy::~CrossExchangeArbitrageStrategy()
{

    for (auto &[_, info] : book_check_info_)
    {
        if (info != nullptr)
        {
            delete info;
            info = nullptr;
        }
    }
    book_check_info_.clear();
}
}  // namespace alphaone
