#include "MultiBookManager.h"

#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/util/String.h"
#include "infrastructure/platform/engine/Engine.h"

namespace alphaone
{
MultiBookManager::MultiBookManager(const GlobalConfiguration *configuration,
                                   const SymbolManager *symbol_manager, Engine *engine,
                                   OrderFactory *order_factory, LevelFactory *level_factory,
                                   const DataSourceType &type)
    : configuration_{configuration}
    , symbol_manager_{symbol_manager}
    , engine_{engine}
    , order_factory_{order_factory}
    , level_factory_{level_factory}
    , is_instantiated_{false}
    , is_locked_{false}
    , default_type_{static_cast<size_t>(type)}
{
    if (engine_ == nullptr)
        throw std::invalid_argument("Engine cannot be nullptr");

    for (auto item : configuration_->GetJson().at("Universe"))
    {
        SPDLOG_INFO("add symbol 123");
        const Symbol *symbol{symbol_manager->GetSymbolByString(item.get<std::string>())};
        SPDLOG_INFO("{}", item.get<std::string>());
        AddSymbol(symbol);
    }

    Init(type);
}

MultiBookManager::~MultiBookManager()
{
}

void MultiBookManager::AddSymbolToUniverse(const Symbol *symbol, const DataSourceType &type)
{
    AddSymbol(symbol);
    AddBook(symbol, type);
}

void MultiBookManager::AddPreBookListener(const Book *book, BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    for (auto &handled_book : handled_books_)
    {
        if (book == handled_book.get())
        {
            handled_book->AddPreBookListener(listener);
        }
    }
}

void MultiBookManager::AddPostBookListener(const Book *book, BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    for (auto &handled_book : handled_books_)
    {
        if (book == handled_book.get())
        {
            handled_book->AddPostBookListener(listener);
        }
    }
}

void MultiBookManager::AddPreBookListener(const Symbol *symbol, BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    bool finded = false;
    for (auto &book : handled_books_)
    {
        if (symbol == book->GetSymbol())
        {
            book->AddPreBookListener(listener);
            finded = true;
        }
    }

    if (BRANCH_UNLIKELY(!finded))
    {
        if (symbol != nullptr)
        {
            SPDLOG_ERROR("cannot find {} in MultiBookManager", symbol->to_string());
        }
        else
        {
            SPDLOG_ERROR("cannot find nullptr in MultiBookManager");
        }
        abort();
    }
}

void MultiBookManager::AddPostBookListener(const Symbol *symbol, BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    bool finded = false;
    for (auto &book : handled_books_)
    {
        if (symbol == book->GetSymbol())
        {
            book->AddPostBookListener(listener);
            finded = true;
        }
    }

    if (BRANCH_UNLIKELY(!finded))
    {
        SPDLOG_ERROR("cannot find {} in MultiBookManager", (void *)symbol);
        abort();
    }
}

void MultiBookManager::AddPreBookListener(BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    for (auto &book : handled_books_)
    {
        book->AddPreBookListener(listener);
    }
}

void MultiBookManager::AddPostBookListener(BookDataListener *listener)
{
    if (BRANCH_UNLIKELY(!IsInstantiated()))
    {
        SPDLOG_ERROR("MultiBookManager has not yet been instantiated");
        abort();
    }

    for (auto &book : handled_books_)
    {
        book->AddPostBookListener(listener);
    }
}

void MultiBookManager::Lock()
{
    is_locked_ = true;
}

void MultiBookManager::OnMarketDataMessage(const MarketDataMessage *mm, void *raw_packet)
{
    for (auto &book : handled_books_)
    {
        if (mm->symbol == book->GetSymbol())
        {
            book->OnMarketDataMessage(mm, raw_packet);
        }
    }
}

bool MultiBookManager::IsInstantiated() const
{
    return is_instantiated_;
}

bool MultiBookManager::IsLocked() const
{
    return is_locked_;
}

bool MultiBookManager::IsInUniverse(const Symbol *symbol) const
{
    return std::find(universe_.cbegin(), universe_.cend(), symbol) != universe_.end();
}

Book *MultiBookManager::GetBook(const std::string &symbol, const DataSourceType type) const
{
    return GetBook(symbol_manager_->GetSymbolByString(symbol), type);
}

Book *MultiBookManager::GetBook(const Symbol *symbol, const DataSourceType type) const
{
    const auto &symbol_to_book =
        source_type_unordered_map_from_symbol_to_book_[type == DataSourceType::Invalid
                                                           ? default_type_
                                                           : static_cast<size_t>(type)];
    if (symbol != nullptr)
    {
        if (auto bit = symbol_to_book.find(symbol); bit != symbol_to_book.end())
        {
            return bit->second.get();
        }
        else
        {
            SPDLOG_WARN("[{}] {} is not in universe", __func__, (*symbol));
            return nullptr;
        }
    }
    else
    {
        SPDLOG_WARN("[{}] symbol is nullptr.", __func__);
        return nullptr;
    }
}

const std::vector<const Symbol *> &MultiBookManager::GetUniverse() const
{
    return universe_;
}

void MultiBookManager::Init(const DataSourceType &type)
{
    SPDLOG_INFO("MultiBookManager::Init");
    source_type_unordered_map_from_symbol_to_book_.resize(static_cast<size_t>(DataSourceType::END));

    for (auto &symbol : GetUniverse())
    {
        SPDLOG_INFO("MultiBookManager::Init Addbook ");
        AddBook(symbol, type);
    }

    size_t book_count = 0;

    for (auto &m : source_type_unordered_map_from_symbol_to_book_)
    {
        book_count += m.size();
    }

    if (universe_.size() != book_count)
    {
        SPDLOG_ERROR("[{}] universe and unordered_map_from_symbol_to_book mismatch", __func__);
        abort();
    }

    is_instantiated_ = true;
}

void MultiBookManager::AddSymbol(const Symbol *symbol)
{
    if (BRANCH_LIKELY(symbol != nullptr))
    {
        if (!IsInUniverse(symbol))
        {
            SPDLOG_INFO("[{}] Add {} into Universe", __func__, symbol->to_string());
            const auto &data_source_id{symbol->GetDataSourceID()};
            SPDLOG_INFO("fkfkfkfkfkfk {}", data_source_id == DataSourceID::BINANCE_PERP);
            SPDLOG_INFO("fkfkfkfkfkfk {}", data_source_id == DataSourceID::TAIFEX_FUTURE);
            universe_.push_back(symbol);
        }
        else
        {
            SPDLOG_WARN("[{}] {} is already in Universe", __func__, symbol->to_string());
            return;
        }
    }
    else
    {
        SPDLOG_ERROR("[{}] symbol is nullptr", __func__);
        abort();
    }
}

void MultiBookManager::AddBook(const Symbol *symbol, const DataSourceType type, void *structure)
{
    SPDLOG_INFO("Add Book for symbol={} and type={}", symbol->to_string(), FromDataSourceTypeToString(type));
    if (symbol == nullptr)
        throw std::invalid_argument(fmt::format("[{}] Add Book for symbol=nullptr", __func__));

    if (IsLocked())
    {
        SPDLOG_WARN("try to add book for symbol={} and type={} after manager has been locked",
                    symbol->to_string(), FromDataSourceTypeToString(type));
        return;
    }

    auto &symbol_to_book{source_type_unordered_map_from_symbol_to_book_[static_cast<size_t>(type)]};
    if (symbol_to_book.find(symbol) != symbol_to_book.cend())
    {
        SPDLOG_WARN("[{}] book for symbol={} and type={} already exists", __func__,
                    symbol->to_string(), FromDataSourceTypeToString(type));
        return;
    }

    if (type == DataSourceType::MarketByOrder)
    {
        SPDLOG_INFO("am i here?11");
        auto om = engine_->GetObjectManager();
        if (symbol->GetDataSourceID() == DataSourceID::TWSE_DATA_FILE ||
            !om->GetMarketDataPath(ProviderID::TWSE_DATA_FILE).first.empty())
        {
            SPDLOG_INFO("[{}] Add TWSEFileBook for {}", __func__, symbol->to_string());
            handled_books_.emplace_back(
                std::make_shared<TWSEFileBook>(symbol, order_factory_, level_factory_));
        }
        else
        {
            SPDLOG_INFO("[{}] Add MarketByOrderBook for {}", __func__, symbol->to_string());
            handled_books_.emplace_back(std::make_shared<MarketByOrderBook>(
                configuration_, symbol, order_factory_, level_factory_));
        }
    }
    else if (type == DataSourceType::MarketByPrice)
    {
        SPDLOG_INFO("[{}] Add MarketByPriceBook for {}", __func__, symbol->to_string());
        handled_books_.emplace_back(std::make_shared<MarketByPriceBook>(symbol));
    }
    else
    {
        throw std::invalid_argument(fmt::format("[{}] Fail to add Book for type={}", __func__,
                                                FromDataSourceTypeToString(type)));
    }

    symbol_to_book.emplace(symbol, handled_books_.back());
    engine_->AddMarketDataListener(symbol, handled_books_.back().get());
}

}  // namespace alphaone
