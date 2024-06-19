#include "Main.h"

namespace alphaone
{
Main::Main(const boost::program_options::variables_map &vm)
    : vm_{vm}
    , is_production_{vm_.count("production") > 0}
    , type_{is_production_ ? EngineEventLoopType::Production : EngineEventLoopType::Simulation}
    , date_{is_production_ ? Date::today() : Date::from_yyyymmdd(vm_["date"].as<int32_t>())}
    , affinity_{0}
    , global_configuration_{nullptr}
    , symbol_manager_{nullptr}
    , object_manager_{nullptr}
    , order_factory_{nullptr}
    , level_factory_{nullptr}
    , multi_book_manager_{nullptr}
    , multi_counter_manager_{nullptr}
    , engine_{nullptr}
    // , command_relay_{nullptr}
    , strategy_factory_{nullptr}
    , strategy_{nullptr}
    , rb_manager_{nullptr}
{
    // set up log level
    spdlog::set_level(static_cast<spdlog::level::level_enum>(vm_["log_level"].as<int>()));

    // set up configurations
    if (!vm_["configuration"].empty())
    {
        configurations_ = vm_["configuration"].as<std::vector<std::string>>();
    }

    // set up environments
    InstallEnvironments();
    std::cout << 1 << std::endl;
    // set up object manager
    InstallObjectManager();
    std::cout << 2 << std::endl;
    // set up engine
    InstallEngine();
    std::cout << 3 << std::endl;
    // set up multi-book manager
    InstallMultiBookManager();
    std::cout << 4 << std::endl;
    // set up multi-counter manager
    InstallMultiCounterManager();
    std::cout << 5 << std::endl;
    // set up affinity
    InstallAffinity();
    std::cout << 6 << std::endl;
    // set up order manager
    InstallOrderManager();
    std::cout << 7 << std::endl;
    // set up strategy
    InstallStrategy();
    std::cout << 8 << std::endl;
    // set up book data listeners
    InstallBookDataListeners();
    std::cout << 9 << std::endl;
}

Main::~Main()
{
    if (strategy_factory_)
    {
        delete strategy_factory_;
        strategy_factory_ = nullptr;
    }
    if (strategy_)
    {
        delete strategy_;
        strategy_ = nullptr;
    }
    for (auto &order_manager : taifex_order_managers_)
    {
        if (order_manager)
        {
            delete order_manager;
            order_manager = nullptr;
        }
    }
    if (multi_counter_manager_)
    {
        delete multi_counter_manager_;
        multi_counter_manager_ = nullptr;
    }
    if (multi_book_manager_)
    {
        delete multi_book_manager_;
        multi_book_manager_ = nullptr;
    }
    if (level_factory_)
    {
        delete level_factory_;
        level_factory_ = nullptr;
    }
    if (order_factory_)
    {
        delete order_factory_;
        order_factory_ = nullptr;
    }
    if (engine_)
    {
        delete engine_;
        engine_ = nullptr;
    }
    if (object_manager_)
    {
        delete object_manager_;
        object_manager_ = nullptr;
    }
}

void Main::InstallEnvironments()
{
    if (is_production_)
    {
        // production only one configuration now
        assert(configurations_.size() == 1);
    }
}

void Main::InstallObjectManager()
{
    // set up object manager
    object_manager_ = new ObjectManager{date_, configurations_};
    std::cout << 6 << std::endl;
    // set up configuration
    global_configuration_ = object_manager_->GetGlobalConfiguration();
    std::cout << 7 << std::endl;
    // set up symbol manager
    symbol_manager_ = object_manager_->GetSymbolManager();
    std::cout << 8 << std::endl;
}

void Main::InstallEngine()
{
    // set up engine
    engine_ = new Engine{object_manager_, date_, type_};

    if (!is_production_)
        return;

    if (global_configuration_->GetJson().contains("RingBufferManager"))
    {
        rb_manager_ = new RingBufferManager{};
        rb_manager_->ReserveBuffer(20);  // may need to adjust
        const auto &rbs = global_configuration_->GetJson()["RingBufferManager"];
        for (const auto &rb : rbs)
        {
            const auto &index  = rb.at("index").get<int>();
            const auto &block  = rb.at("block").get<int>();
            const auto &packet = rb.at("packet").get<int>();
            const auto &length = rb.at("length").get<int>();
            rb_manager_->AddRingBuffer(index, block, packet, length);
        }

        engine_->SetRingBufferManager(rb_manager_);
    }
}

void Main::InstallMultiBookManager()
{
    // set up multi-book manager
    SPDLOG_INFO("Main::InstallMultiBookManager 1");
    order_factory_      = new OrderFactory{};
    SPDLOG_INFO("Main::InstallMultiBookManager 2");
    level_factory_      = new LevelFactory{};
    SPDLOG_INFO("Main::InstallMultiBookManager 3");
    multi_book_manager_ = new MultiBookManager{global_configuration_, symbol_manager_, engine_,
                                               order_factory_, level_factory_};
}

void Main::InstallMultiCounterManager()
{
    // set up multi-counter manager
    multi_counter_manager_ = new MultiCounterManager{object_manager_, multi_book_manager_, engine_};
}

void Main::InstallAffinity()
{
    if (is_production_)
    {
        // set up affinity
        if (global_configuration_->GetJson().contains("/System/affinity"_json_pointer))
        {
            affinity_ = global_configuration_->GetJson()["System"]["affinity"].get<int>();
        }
        else
        {
            SPDLOG_ERROR("[\"System\"][\"affinity\"] is not specified in global configuration, "
                         "going to bind at core 0");
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(affinity_, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
}

void Main::InstallOrderManager()
{
    // set up order manager
    if (!is_production_)
    {
        for (size_t config_index = 0; config_index < configurations_.size(); ++config_index)
        {
            taifex_order_managers_.push_back(new TaifexSimulationOrderManager(
                multi_book_manager_, object_manager_, engine_, config_index));
        }
    }
    else
    {
        if (global_configuration_->GetJson().contains("OrderSource"))
        {
            if (global_configuration_->GetJson()["OrderSource"].contains("Taifex"))
            {
                taifex_order_managers_.push_back(new TaifexOrderManager{object_manager_});
            }
        }
    }

    for (auto &order_manager : taifex_order_managers_)
    {
        engine_->AddOrderManager(order_manager);
    }
}

std::string Main::GetStrategyName(size_t config_index) const
{
    std::cout << 10 << std::endl;
    const auto &node(
        object_manager_->GetGlobalConfiguration(config_index)->GetJson().at("Strategy"));
    std::cout << 11 << std::endl;
    if (node.contains("strategy"))
    {
        std::cout << 3232 << std::endl;
        return node["strategy"].get<std::string>();
    }
    else
    {
        std::cout << 666 << std::endl;
        if (node.size() != 1)
        {
            SPDLOG_WARN("more than one strategy is assigned in process. only the first one will be "
                        "instantiated.");
        }
        std::cout << 123 << std::endl;
        for (auto &item : node.items())
        {
            std::cout << 324234 << std::endl;
            return item.key();
        }
        return "";
    }
}

void Main::InstallStrategy()
{
    // retrieve strategy from configuration
    if (is_production_)
    {
        auto tx_manager = taifex_order_managers_.empty() ? nullptr : taifex_order_managers_[0];
        const auto &strategy_name = GetStrategyName(0);
        FileLock    file_lock{strategy_name, affinity_};
        strategy_ = CreateStrategy(strategy_name, object_manager_, multi_book_manager_,
                                   multi_counter_manager_, engine_, tx_manager);
        if (!strategy_)
        {
            SPDLOG_ERROR("strategy is nullptr");
            abort();
        }

        if (tx_manager)
        {
            tx_manager->AddOrderReportListener(strategy_);
        }

        if (rb_manager_)
        {
            rb_manager_->AddRingBufferListener(strategy_);
        }

        // set up command subscription
        if (const auto sys{"System"}; global_configuration_->GetJson().contains(sys))
        {
            if (const auto server{"nats_server"};
                global_configuration_->GetJson()[sys].contains(server))
            {
                // command_relay_->Connect(
                //     global_configuration_->GetJson()[sys][server].get<std::string>());
            }
            else
            {
                SPDLOG_ERROR("When Production \"{}\" is a mandatory field in config[\"{}\"]",
                             server, sys);
                throw std::invalid_argument("wrong config format");
            }

            if (const auto port{"command_port"};
                global_configuration_->GetJson()[sys].contains(port))
            {
                // command_relay_->Subscribe(
                //     global_configuration_->GetJson()[sys][port].get<std::string>());
            }
            else
            {
                SPDLOG_ERROR("When Production \"{}\" is a mandatory field in config[\"{}\"]", port,
                             sys);
                throw std::invalid_argument("wrong config format");
            }
        }
        else
        {
            SPDLOG_ERROR("When Production \"{}\" is a mandatory field in config", sys);
            throw std::invalid_argument("wrong config format");
        }

        // set up command relay and command listeners
        // command_relay_->AddCommandListener(strategy_);
        // engine_->AddCommandRelay(command_relay_);
    }
    else // simulation
    {
        std::cout << 888 << std::endl;
        strategy_factory_ = new StrategyFactory(
            object_manager_, multi_book_manager_, multi_counter_manager_, engine_,
            &taifex_order_managers_);
        std::cout << 999 << std::endl;
        for (size_t i{0}; i < taifex_order_managers_.size(); ++i)
        {
            std::cout << 45678765 << std::endl;
            Strategy *strategy{strategy_factory_->CreateStrategy(GetStrategyName(i), i)};
            std::cout << 365456543 << std::endl;
            if (!strategy)
            {
                SPDLOG_WARN("strategy={} is nullptr and thus skipped", GetStrategyName(i));
                continue;
            }

            taifex_order_managers_[i]->AddOrderReportListener(strategy);
            std::cout << 333 << std::endl;
            for (auto &ensemble : *(strategy->GetEnsembles()))
            {
                for (auto &tactic : *(ensemble->GetAllTactics()))
                {
                    tactic->TurnOn();
                }
            }
            strategy->TurnOn();

            strategies_.push_back(strategy);
        }
    }
}

void Main::InstallBookDataListeners()
{

    if (strategy_)
    {
        // add strategy to pre book listeners
        multi_book_manager_->AddPreBookListener(strategy_);

        // add strategy to post book listeners
        multi_book_manager_->AddPostBookListener(strategy_);
    }

    for (auto &strategy : strategies_)
    {
        // add strategy to pre book listeners
        multi_book_manager_->AddPreBookListener(strategy);

        // add strategy to post book listeners
        multi_book_manager_->AddPostBookListener(strategy);
    }

    // add simulators to post book listeners (simulators should be the last book listeners)
    if (!is_production_)
    {
        for (auto &taifex_order_manager : taifex_order_managers_)
        {
            dynamic_cast<TaifexSimulationOrderManager *>(taifex_order_manager)
                ->SetUpSimulatorListeners();
        }
    }
}

void Main::Execute()
{
    // run event loop
    engine_->RunEventLoop();
}

}  // namespace alphaone
