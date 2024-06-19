#include "CounterFactory.h"

// core counters
#include "infrastructure/common/util/String.h"
#include "infrastructure/platform/counter/AddIntervalCounter.h"
#include "infrastructure/platform/counter/AxisIntervalCounter.h"
#include "infrastructure/platform/counter/DeleteIntervalCounter.h"
#include "infrastructure/platform/counter/DoubleTickIntervalCounter.h"
#include "infrastructure/platform/counter/HalfTickIntervalCounter.h"
#include "infrastructure/platform/counter/MessageIntervalCounter.h"
#include "infrastructure/platform/counter/SingleTickIntervalCounter.h"
#include "infrastructure/platform/counter/TimeIntervalCounter.h"
#include "infrastructure/platform/counter/TouchIntervalCounter.h"
#include "infrastructure/platform/counter/TradeIntervalCounter.h"
#include "infrastructure/platform/counter/TradeQtyIntervalCounter.h"

// customized counters
#include "counter/AdjustedWeightedTickIntervalCounter.h"
#include "counter/ClockTimeIntervalCounter.h"
#include "counter/EventIntervalCounter.h"
#include "counter/TradeMoneyIntervalCounter.h"
#include "counter/TradeThroughIntervalCounter.h"
#include "counter/TrueRangeIntervalCounter.h"
#include "counter/WeightedTickIntervalCounter.h"

#include <boost/algorithm/string.hpp>

namespace alphaone
{
Counter *CounterFactory::RetrieveCounterFromCounterSpec(const ObjectManager *object_manager,
                                                        MultiBookManager *   multi_book_manager,
                                                        Engine *engine, const std::string &interval,
                                                        const nlohmann::json &spec)
{
    const auto &split_interval = SplitIntoVector(interval, "_", true);
    const auto &symbol_str     = spec.value("symbol", "");
    const auto &symbol         = object_manager->GetSymbolManager()->GetSymbolByString(symbol_str);
    const auto &book           = multi_book_manager->GetBook(symbol);

    // core counter
    if (split_interval[0] == "SingleTickInterval")
    {
        return new SingleTickIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "DoubleTickInterval")
    {
        return new DoubleTickIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "HalfTickInterval")
    {
        return new HalfTickIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "AxisInterval")
    {
        return new AxisIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "TradeInterval")
    {
        return new TradeIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "MessageInterval")
    {
        return new MessageIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "AddInterval")
    {
        return new AddIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "DeleteInterval")
    {
        return new DeleteIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "TradeThroughInterval")
    {
        return new TradeThroughIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "TouchInterval")
    {
        return new TouchIntervalCounter(book, multi_book_manager);
    }
    if (split_interval[0] == "EventInterval")
    {
        return new EventIntervalCounter(book, multi_book_manager);
    }

    auto        copy_spec  = spec;
    const auto &split_size = split_interval.size();
    if (split_interval[0] == "WeightedTickInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["price_type"] = split_interval[1];
        }
        if (split_size >= 3UL)
        {
            copy_spec["tick"] = std::stod(split_interval[2]);
        }
        return new WeightedTickIntervalCounter(book, multi_book_manager, copy_spec);
    }
    if (split_interval[0] == "TradeQtyInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["lots"] = std::stod(split_interval[1]);
        }
        return new TradeQtyIntervalCounter(book, multi_book_manager, copy_spec);
    }
    if (split_interval[0] == "TradeMoneyInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["amount"] = std::stod(split_interval[1]);
        }
        return new TradeMoneyIntervalCounter(object_manager, multi_book_manager, copy_spec);
    }
    if (split_interval[0] == "TimeInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["duration_in_second"] = std::stod(split_interval[1]);
        }
        return new TimeIntervalCounter(book, multi_book_manager, copy_spec);
    }

    // customized counters
    if (split_interval[0] == "AdjustedWeightedTickInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["multiplier"] = std::stod(split_interval[1]);
        }
        return new AdjustedWeightedTickIntervalCounter(book, multi_book_manager, copy_spec);
    }
    if (split_interval[0] == "ClockTimeInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["duration"] = std::stod(split_interval[1]);
        }
        if (split_size >= 3UL)
        {
            copy_spec["time_start"] = split_interval[2];
        }
        if (split_size >= 4UL)
        {
            copy_spec["time_end"] = split_interval[3];
        }
        return new ClockTimeIntervalCounter(book, multi_book_manager, engine, copy_spec);
    }
    if (split_interval[0] == "TrueRangeInterval")
    {
        if (split_size >= 2UL)
        {
            copy_spec["tick"] = std::stod(split_interval[1]);
        }
        if (split_size >= 3UL)
        {
            copy_spec["ClockTimeInterval"]["duration"] = std::stod(split_interval[2]);
        }
        if (split_size >= 4UL)
        {
            copy_spec["ClockTimeInterval"]["time_start"] = split_interval[3];
        }
        if (split_size >= 5UL)
        {
            copy_spec["ClockTimeInterval"]["time_end"] = split_interval[4];
        }
        return new TrueRangeIntervalCounter(object_manager, multi_book_manager, engine, copy_spec);
    }

    return nullptr;
}
}  // namespace alphaone
