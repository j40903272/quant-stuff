#ifndef _CROSSEXCHANGEARBITRAGESTRATEGY_H_
#define _CROSSEXCHANGEARBITRAGESTRATEGY_H_

#include "infrastructure/common/numeric/SmoothNumber.h"
#include "infrastructure/common/util/Order.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/metric/MetricReporter.h"
#include "risk/IdiotProof.h"
#include "risk/RiskController.h"
#include "strategy/FeedPriceChecker.h"
#include "strategy/MarkoutTrackRecord.h"
#include "strategy/Strategy.h"
#include "strategy/StrategyFactory.h"
#include "strategy/TurnOverPeriod.h"

#include <mutex>
#include <unordered_set>

namespace alphaone
{
class CrossExchangeArbitrageStrategy;

class CrossExchangeArbitrageStrategy : public Strategy
{
  public:
    CrossExchangeArbitrageStrategy(ObjectManager *object_manager, MultiBookManager *multi_book_manager,
                 MultiCounterManager *multi_counter_manager, Engine *engine,
                 TaifexOrderManagerBase *order_manager);
    CrossExchangeArbitrageStrategy(const CrossExchangeArbitrageStrategy &) = delete;
    CrossExchangeArbitrageStrategy &operator=(const CrossExchangeArbitrageStrategy &) = delete;

    void OnPacketEnd(const Timestamp event_loop_time, const BookDataMessagePacketEnd *o) final;

    ~CrossExchangeArbitrageStrategy();

  private:
    std::unordered_map<const Book *, BookCheckInfo *> book_check_info_;
};
}  // namespace alphaone

#endif
