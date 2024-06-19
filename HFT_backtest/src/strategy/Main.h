#ifndef _STRATEGYMAIN_H_
#define _STRATEGYMAIN_H_

#include "infrastructure/common/typedef/Typedefs.h"
#include "infrastructure/common/util/FileLock.h"
#include "infrastructure/platform/engine/Engine.h"
#include "infrastructure/platform/manager/MultiBookManager.h"
#include "infrastructure/platform/manager/SymbolManager.h"
#include "infrastructure/platform/simulator/TaifexSimulator.h"
#include "strategy/StrategyFactory.h"
#include "status/Version.h"

#include <boost/program_options.hpp>
#include <filesystem>

namespace alphaone
{
class Main
{
  public:
    Main(const boost::program_options::variables_map &vm);
    Main()             = delete;
    Main(const Main &) = delete;
    Main &operator=(const Main &) = delete;
    Main(Main &&)                 = delete;
    Main &operator=(Main &&) = delete;

    ~Main();

    void Execute();

  private:
    void InstallEnvironments();
    void InstallObjectManager();
    void InstallEngine();
    void InstallMultiBookManager();
    void InstallMultiCounterManager();
    void InstallAffinity();
    void InstallOrderManager();
    void InstallStrategy();
    void InstallBookDataListeners();

    std::string GetStrategyName(size_t config_index) const;

    // environments
    const boost::program_options::variables_map &vm_;
    const bool                                   is_production_;
    const EngineEventLoopType                    type_;
    const Date                                   date_;
    int                                          affinity_;

    // multiple
    std::vector<std::string> configurations_;

    // platforms
    GlobalConfiguration *                 global_configuration_;
    SymbolManager *                       symbol_manager_;
    ObjectManager *                       object_manager_;
    OrderFactory *                        order_factory_;
    LevelFactory *                        level_factory_;
    MultiBookManager *                    multi_book_manager_;
    MultiCounterManager *                 multi_counter_manager_;
    Engine *                              engine_;
    std::vector<TaifexOrderManagerBase *> taifex_order_managers_;
    StrategyFactory *                     strategy_factory_;
    Strategy *                            strategy_;    // production
    std::vector<Strategy *>               strategies_;  // simulation
    RingBufferManager *                   rb_manager_;
    nlohmann::json                        new_fit_json_;
};
}  // namespace alphaone

#endif
