#include "Strategy.h"

namespace alphaone
{
Tactic::Tactic(const std::string &name, Ensemble *ensemble)
    : name_{name}
    , id_{ensemble->GenerateID(name)}
    , config_{ensemble->GetStrategy()->GetConfiguration()}
    , symbol_{ensemble->GetSymbol()}
    , book_{ensemble->GetBook()}
    , position_{config_, symbol_, book_}
    , ensemble_{ensemble}
    , is_on_{false}
{
}

Tactic::Tactic(const std::string &name, const GlobalConfiguration *config, const Symbol *symbol,
               const Book *book, const size_t id)
    : name_{name}
    , id_{id}
    , config_{config}
    , symbol_{symbol}
    , book_{book}
    , position_{config_, symbol_, book_}
    , ensemble_{nullptr}
    , is_on_{false}
{
}

Tactic::Tactic(const std::string &name, Ensemble *ensemble, const nlohmann::json &fee)
    : name_{name}
    , id_{ensemble->GenerateID(name)}
    , config_{ensemble->GetStrategy()->GetConfiguration()}
    , symbol_{ensemble->GetSymbol()}
    , book_{ensemble->GetBook()}
    , position_{symbol_, book_, fee.at("fee_rate").get<double>(), fee.at("fee_cost").get<double>(),
                fee.at("tax_rate").get<double>()}
    , ensemble_{ensemble}
    , is_on_{false}
{
}

void Tactic::RegisterSymbol(const std::string &symbol, const DataSourceType &type)
{
    GetEnsemble()->RegisterSymbol(symbol, type);
}

void Tactic::RegisterSymbol(const Symbol *symbol, const DataSourceType &type)
{
    GetEnsemble()->RegisterSymbol(symbol, type);
}

void Tactic::OnActivated(const Timestamp event_loop_time)
{
    if (!IsSimulation())
    {
        std::stringstream ss;
        ss << "Turn " << GetName() << " of " << GetSymbol()->to_string() << " On";
        GetEnsemble()->GetStrategy()->Response(ss.str());
    }
}

void Tactic::OnDeactivated(const Timestamp event_loop_time)
{
    if (!IsSimulation())
    {
        std::stringstream ss;
        ss << "Turn " << GetName() << " of " << GetSymbol()->to_string() << " Off";
        GetEnsemble()->GetStrategy()->Response(ss.str());
    }
}

void Tactic::UpdatePosition(const Timestamp event_loop_time, const Symbol *symbol,
                            const BookSide side, const int price, const int qty)
{
    if (symbol != symbol_)
    {
        return;
    }

    GetEnsemble()->UpdatePosition(event_loop_time, symbol, side, price, qty);
}

bool Tactic::IsSimulation() const
{
    return GetEnsemble()->GetStrategy()->GetEngine()->IsSimulation();
}

Ensemble::Ensemble(const std::string &name, const Symbol *symbol, Strategy *strategy)
    : name_{name}
    , config_{strategy->GetConfiguration()}
    , symbol_{symbol}
    , book_{strategy->GetMultiBookManager() ? strategy->GetMultiBookManager()->GetBook(symbol)
                                            : nullptr}
    , position_{config_, symbol_, book_}
    , strategy_{strategy}
{
}

Ensemble::Ensemble(const std::string &name, const GlobalConfiguration *config, const Symbol *symbol,
                   const Book *book)
    : name_{name}
    , config_{config}
    , symbol_{symbol}
    , book_{book}
    , position_{config_, symbol_, book_}
    , strategy_{nullptr}
{
}

Ensemble::Ensemble(const std::string &name, const Symbol *symbol, Strategy *strategy,
                   const nlohmann::json &fee)
    : name_{name}
    , config_{strategy->GetConfiguration()}
    , symbol_{symbol}
    , book_{strategy->GetMultiBookManager() ? strategy->GetMultiBookManager()->GetBook(symbol)
                                            : nullptr}
    , position_{symbol_, book_, fee.at("fee_rate").get<double>(), fee.at("fee_cost").get<double>(),
                fee.at("tax_rate").get<double>()}
    , strategy_{strategy}
{
}

void Ensemble::RegisterSymbol(const std::string &symbol, const DataSourceType &type)
{
    GetStrategy()->RegisterSymbol(symbol, type);
}

void Ensemble::RegisterSymbol(const Symbol *symbol, const DataSourceType &type)
{
    GetStrategy()->RegisterSymbol(symbol, type);
}


void Ensemble::UpdatePosition(const Timestamp event_loop_time, const Symbol *symbol,
                              const BookSide side, const int price, const int qty)
{
    if (symbol != symbol_)
    {
        return;
    }

    position_.Update(side, price, qty);

    for (auto &tactic : tactics_)
    {
        tactic->GetPosition()->Update(side, price, qty);
    }
}

bool Ensemble::IsSimulation() const
{
    return GetStrategy()->GetEngine()->IsSimulation();
}

Strategy::Strategy(const std::string &name, ObjectManager *object_manager,
                   MultiBookManager *multi_book_manager, MultiCounterManager *multi_counter_manager,
                   Engine *engine, TaifexOrderManagerBase *taifex_order_manager)
    : name_{name}
    , object_manager_{object_manager}
    , config_{object_manager->GetGlobalConfiguration(0)}
    , symbol_manager_{object_manager->GetSymbolManager()}
    , multi_book_manager_{multi_book_manager}
    , multi_counter_manager_{multi_counter_manager}
    , engine_{engine}
    , taifex_order_manager_{taifex_order_manager}
    , heartbeat_string_{config_->GetJson().contains("/Strategy/HeartbeatString"_json_pointer)
                            ? config_->GetJson()["Strategy"]["HeartbeatString"].get<std::string>()
                            : "Ping"}
    , last_heartbeat_ts_{Timestamp::invalid()}
    , is_on_{false}
    , packet_end_count_{0}
    , registered_symbol_{nullptr}
    , switched_tactic_{nullptr}
    , mm_{DataSourceType::Invalid}
{
    std::cout << 333333333333 << std::endl;
}

Strategy::~Strategy()
{
    if (!IsSimulation())
    {
        Response("EXIT");
    }
}

// void Strategy::OnCommand(const Timestamp &event_loop_time, const MessageFormat::GenericMessage &gm)
// {
//     auto wit = gm.msgdata().stringmap().find("word");
//     if (wit == gm.msgdata().stringmap().end())
//         return;
//     OnCommandHelper(event_loop_time, wit->second);
// }

void Strategy::OnCommandHelper(const Timestamp &event_loop_time, const std::string &word)
{
    UpdateLastHeartbeat(event_loop_time);

    if (word.find(heartbeat_string_) == std::string::npos)
    {
        SPDLOG_INFO("{} received \"{}\"", event_loop_time, word);
    }
    else
    {
        return;
    }
    const auto &action = boost::to_upper_copy(word);
    if (action == "HELP")
    {
        std::stringstream ss;
        // clang-format off
            ss << "\n";
            ss << "[USER MANUAL]\n";
            ss << "\"\"                       -- print book\n";
            ss << "\"STATUS\"                 -- print status\n";
            ss << "\"SYMBOL\"                 -- print symbols in universe\n";
            ss << "\"ITEM\"                   -- print registered symbol and switched tactic\n";
            ss << "\"LIST\"                   -- print all symbols and all tactics\n";
            ss << "\"ON\"                     -- turn tactic on\n";
            ss << "\"OFF\"                    -- turn tactic off\n";
            ss << "\"DONE\"                   -- terminate process\n";
            ss << "\"FLAT\"                   -- flatten position\n";
            ss << "\"REGISTER SYMBOLID\"      -- register symbol\n";
            ss << "\"UNREGISTER\"             -- unregister symbol\n";
            ss << "\"SWITCH TACTICID\"        -- switch tactic\n";
            ss << "\"UNSWITCH\"               -- unswitch tactic\n";
            ss << "\"B @PRICE #QTY\"          -- send bid order at price of qty for registered symbol and switched tactic\n";
            ss << "\"S @PRICE #QTY\"          -- send ask order at price of qty for registered symbol and switched tactic\n";
        // clang-format on

        Response(ss.str());

        return;
    }

    if (action == "")
    {
        if (registered_symbol_ == nullptr)
        {
            for (auto &symbol : this->GetMultiBookManager()->GetUniverse())
            {
                Response(this->GetMultiBookManager()->GetBook(symbol)->DumpString());
            }
            return;
        }

        for (auto &ensemble : *this->GetEnsembles())
        {
            if (registered_symbol_ == ensemble->GetSymbol())
            {
                Response(ensemble->GetBook()->DumpString());
            }
        }
        return;
    }

    if (action == "STATUS")
    {
        nlohmann::json status;

        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                status[tactic->GetSymbol()->to_string()].push_back(tactic->OnCommandDumpStatus());
            }
        }

        std::stringstream ss;
        ss << '\n' << status.dump(4, ' ');
        Response(ss.str());

        return;
    }

    if (action == "ITEM")
    {
        std::stringstream ss;
        ss << "\n";
        ss << "Registered symbol: "
           << (registered_symbol_ == nullptr ? "undefine"
                                             : "[" + registered_symbol_->to_string() + "]")
           << "\n";
        ss << "Switched tactic: "
           << (switched_tactic_ == nullptr ? "undefine"
                                           : "[" + switched_tactic_->GetName() + "][" +
                                                 std::to_string(switched_tactic_->GetID()) + "]")
           << "\n";
        Response(ss.str());
        return;
    }

    if (action == "LIST")
    {
        std::stringstream ss;

        size_t symbol_count{0};
        for (auto &ensemble : *this->GetEnsembles())
        {
            ss << "[" << symbol_count << "] symbol: " << ensemble->GetSymbol() << "\n";

            size_t tactic_count{0};
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                ss << "\t[" << tactic_count << "] tactic: [" << tactic->GetName() << "]["
                   << tactic->GetID() << "]\n";
                ++tactic_count;
            }

            ++symbol_count;
        }

        Response(ss.str());

        return;
    }

    if (action == "ON")
    {
        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                tactic->TurnOn(Timestamp::now());
            }
        }
        TurnOn();
        return;
    }

    if (action == "OFF")
    {
        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                tactic->TurnOff(Timestamp::now());
            }
        }
        TurnOff();
        return;
    }

    if (action == "O")
    {
        nlohmann::json status;

        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                status[tactic->GetSymbol()->to_string()].push_back(
                    tactic->OnCommandDumpOutstandingOrder());
            }
        }

        std::stringstream ss;
        ss << '\n' << status.dump(4, ' ');
        Response(ss.str());

        return;
    }

    if (action == "DONE")
    {
        bool is_all_off{true};
        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                if (tactic->IsOn())
                {
                    is_all_off = false;
                }
            }
        }

        if (is_all_off)
        {
            GetEngine()->EndEventLoop();
        }
        else
        {
            Response("all tactics should be turned off before process being terminated");
        }

        return;
    }

    if (action == "FLAT")
    {
        for (auto &ensemble : *this->GetEnsembles())
        {
            for (auto &tactic : *ensemble->GetAllTactics())
            {
                tactic->OnCommandFlat();
            }
        }

        return;
    }

    if (action == "SYMBOL")
    {
        std::stringstream ss;
        ss << "\n";

        size_t symbol_id{0};
        for (auto &ensemble : *this->GetEnsembles())
        {
            ss << "[" << symbol_id << "] " << ensemble->GetSymbol()->to_string();
            ++symbol_id;
        }

        Response(ss.str());

        return;
    }

    const std::vector<std::string> word_split{SplitIntoVector(word, " ", true)};

    if (word_split.size() == 2 && boost::to_upper_copy(word_split[0]) == "REGISTER")
    {
        try
        {
            const int symbol_id{std::stoi(word_split[1])};

            if (symbol_id >= static_cast<int>(GetEnsembles()->size()) or symbol_id < 0)
            {
                std::stringstream ss;
                ss << "symbol given by \"" << word_split[1] << "\" cannot be registered";
                Response(ss.str());
            }
            else
            {
                registered_symbol_ = (*GetEnsembles()).at(symbol_id)->GetSymbol();

                std::stringstream ss;
                ss << "[" << registered_symbol_->to_string() << "] registered";
                Response(ss.str());
            }
        }
        catch (...)
        {
            std::stringstream ss;
            ss << "symbol given by \"" << word_split[1] << "\" cannot be handled";
            Response(ss.str());
        }

        return;
    }

    if (action == "UNREGISTER")
    {
        registered_symbol_ = nullptr;
        Response("unregistered");
        return;
    }

    if (word_split.size() == 2 && boost::to_upper_copy(word_split[0]) == "SWITCH")
    {
        for (auto &ensemble : *this->GetEnsembles())
        {
            if (ensemble->GetSymbol() == registered_symbol_)
            {
                try
                {
                    const int tactic_id{std::stoi(word_split[1])};

                    if (tactic_id >= static_cast<int>(ensemble->GetAllTactics()->size()) or
                        tactic_id < 0)
                    {
                        std::stringstream ss;
                        ss << "tactic given by \"" << word_split[1] << "\" cannot be switched";
                        Response(ss.str());
                    }
                    else
                    {
                        switched_tactic_ = ensemble->GetAllTactics()->at(tactic_id);

                        std::stringstream ss;
                        ss << "[" << switched_tactic_->GetName() << "]["
                           << switched_tactic_->GetID() << "] switched";
                        Response(ss.str());
                    }
                }
                catch (...)
                {
                    std::stringstream ss;
                    ss << "tactic given by \"" << word_split[1] << "\" cannot be handled";
                    Response(ss.str());
                }
            }
        }

        return;
    }

    if (action == "UNSWITCH")
    {
        switched_tactic_ = nullptr;
        Response("unswitched");
        return;
    }

    if (boost::to_upper_copy(word_split[0]) == "B" or boost::to_upper_copy(word_split[0]) == "S")
    {
        const BookSide side{boost::to_upper_copy(word_split[0]) == "B" ? BID : ASK};

        BookPrice price{0};
        BookQty   qty{0};
        for (const auto &item : word_split)
        {
            if (item[0] == '#')
            {
                const std::vector<std::string> qty_splitted{SplitIntoVector(item, "#", true)};
                try
                {
                    if (qty_splitted.size() > 1)
                    {
                        qty = std::stod(qty_splitted[1]);
                    }
                }
                catch (...)
                {
                    std::stringstream ss;
                    ss << "qty given by \"" << qty_splitted[1] << "\" cannot be handled";
                    Response(ss.str());
                }
                continue;
            }

            if (item[0] == '@')
            {
                const std::vector<std::string> price_splitted{SplitIntoVector(item, "@", true)};
                try
                {
                    if (price_splitted.size() > 1)
                    {
                        price = std::stod(price_splitted[1]);
                    }
                }
                catch (...)
                {
                    std::stringstream ss;
                    ss << "price given by \"" << price_splitted[1] << "\" cannot be handled";
                    Response(ss.str());
                }
                continue;
            }
        }

        if (registered_symbol_ != nullptr)
        {
            if (switched_tactic_ != nullptr)
            {
                if (price > 0.0 && qty > 0.0)
                {
                    const nlohmann::json json(
                        switched_tactic_->OnCommandSend(event_loop_time, side, price, qty));

                    if (json.contains("status"))
                    {
                        std::stringstream ss;
                        ss << event_loop_time << " send " << (side == BID ? "BID" : "ASK")
                           << " order price=" << price << " qty=" << qty
                           << " status=" << json["status"] << "\n";
                        Response(ss.str());
                    }

                    return;
                }
                else if (price == 0.0 && qty > 0.0)
                {
                    if (GetMultiBookManager()->GetBook(registered_symbol_)->IsValid())
                    {
                        if (side == BID)
                        {
                            price =
                                GetMultiBookManager()->GetBook(registered_symbol_)->GetPrice(ASK);
                        }
                        else
                        {
                            price =
                                GetMultiBookManager()->GetBook(registered_symbol_)->GetPrice(BID);
                        }

                        const nlohmann::json json(
                            switched_tactic_->OnCommandSend(event_loop_time, side, price, qty));

                        if (json.contains("status"))
                        {
                            std::stringstream ss;
                            ss << event_loop_time << " send " << (side == BID ? "BID" : "ASK")
                               << " order price=" << price << " qty=" << qty
                               << " status=" << json["status"] << "\n";
                            Response(ss.str());
                        }

                        return;
                    }
                }
            }
            else
            {
                Response("switched tactic is nullptr");
                return;
            }
        }
        else
        {
            Response("registered symbol is nullptr");
            return;
        }
    }

    {
        Response("\"" + word + "\" cannot be handled by " + GetName());
        return;
    }
}
}  // namespace alphaone