#include "Symbol.h"
namespace alphaone
{
std::string ProductTypeToString(ProductType type)
{
    switch (type)
    {
    case ProductType::Future:
        // SPDLOG_INFO("[{}] ProductTypeToString: FUTURE", __func__);
        return "FUTURE";
    case ProductType::Option:
        return "OPTION";
    case ProductType::Security:
        return "SECURITY";
    case ProductType::Warrant:
        return "WARRANT";
    case ProductType::Perp:
        return "PERP";
    case ProductType::Spot:
        return "SPOT";
    default:
        return "NONE";
    }
}

ProductType StringToProductType(const std::string &type)
{
    if (type == "FUTURE")
    {
        // SPDLOG_INFO("[{}] ProductType: FUTURE", __func__);
        return ProductType::Future;
    }
    if (type == "SECURITY")
    {
        return ProductType::Security;
    }
    if (type == "PERP")
    {
        return ProductType::Perp;
    }
    return ProductType::None;
}

std::string WarrantTypeToString(WarrantType type)
{
    return "Undefined";
}

WarrantType StringToWarrantType(const std::string &type)
{
    return WarrantType::Undefined;
}

ENUM_CPFlag StringToCPFlag(const std::string &cpflag)
{
    return ENUM_CPFlag::NONE;
}

DataSourceID GetDataSourceID(const std::string &source, ProductType type)
{
    SPDLOG_INFO("[{}] GetDataSourceID {}", __func__, source);
    if (source == "TAIFEX" && type == ProductType::Future)
    {
        return DataSourceID::TAIFEX_FUTURE;
    }
    if (source == "TSE" && (type == ProductType::Security || type == ProductType::Warrant))
    {
        return DataSourceID::TSE;
    }
    if (source == "TWSE_DATA_FILE")
    {
        return DataSourceID::TWSE_DATA_FILE;
    }
    if (source == "BINANCE" && type == ProductType::Perp)
    {
        SPDLOG_INFO("[{}] GetDataSourceID BINANCE_PERP", __func__);
        return DataSourceID::BINANCE_PERP;
    }
    return DataSourceID::UNKNOWN;
}

DataSourceID GetDataSourceID(const std::string &source, const std::string &type_str)
{
    // SPDLOG_INFO("[{}] GetDataSourceID", __func__);
    return GetDataSourceID(source, StringToProductType(type_str));
}

Symbol::Symbol(std::string source, std::string group, std::string pid, std::string cpid,
               std::string underlying, ProductType product_type, WarrantType warrant_type,
               std::string maturitydate, int y, int m, int w, double cr, int v, ENUM_CPFlag cpflag,
               double strike, int multiplier, std::string currency, int dp,
               std::vector<TickSize> &ticksize_list, BookPrice ref_price,
               std::vector<BookPrice> &downlimits, std::vector<BookPrice> &uplimits, bool is_roll,
               const Symbol *c1, const Symbol *c2)
    : source_{source}
    , group_{group}
    , pid_{pid}
    , cpid_{cpid}
    , underlying_{underlying}
    , datasourcepid_{pid.append(std::string(SYMBOLID_LENGTH - pid.size(), ' '))}
    , product_type_{product_type}
    , warrant_type_{warrant_type}
    , maturity_{maturitydate}
    , maturity_date_{maturitydate == "" ? Date::invalid_date()
                                        : Timestamp::from_date_time(maturity_.c_str()).to_date()}
    , y_{y}
    , m_{m}
    , w_{w}
    , conversion_rate_{cr}
    , volume_{v}
    , cpflag_{cpflag}
    , strike_{strike}
    , multiplier_{multiplier}
    , dp_{dp}
    , dc_{std::pow(10, dp)}
    , currency_{currency}
    , ticksize_list_{ticksize_list}
    , name_{"source=" + source_ + "|" + "type=" + ProductTypeToString(product_type_) + "|" +
            (cpid_ != "" ? "cpid=" + cpid_ : "pid=" + pid_)}
    , data_source_id_{alphaone::GetDataSourceID(source, product_type)}
    , ref_price_{ref_price}
    , limits_{downlimits, uplimits}
    , is_roll_{is_roll}
    , c1_{c1}
    , c2_{c2}
{
}

const std::string &Symbol::GetSource() const
{
    return source_;
}

const std::string &Symbol::GetGroup() const
{
    return group_;
}

const std::string &Symbol::GetPid() const
{
    return pid_;
}

const std::string &Symbol::GetCPid() const
{
    return cpid_;
}

const std::string &Symbol::GetUnderlying() const
{
    return underlying_;
}

const std::string &Symbol::GetDataSourcePid() const
{
    return datasourcepid_;
}

const std::string &Symbol::GetRepresentativePid() const
{
    return cpid_ == "" ? pid_ : cpid_;
}

const std::string &Symbol::to_string() const
{
    return name_;
}

const std::string &Symbol::GetMaturity() const
{
    return maturity_;
}

const Date &Symbol::GetMaturityDate() const
{
    return maturity_date_;
}

int Symbol::GetDecimalPrecision() const
{
    return dp_;
}

double Symbol::GetDecimalConverter() const
{
    return dc_;
}

double Symbol::GetConversionRate() const
{
    return conversion_rate_;
}

int Symbol::GetVolume() const
{
    return volume_;
}

int Symbol::GetMultiplier() const
{
    return multiplier_;
}

ENUM_CPFlag Symbol::GetCPFlag() const
{
    return cpflag_;
}

double Symbol::GetStrike() const
{
    return strike_;
}

DataSourceID Symbol::GetDataSourceID() const
{
    SPDLOG_INFO("am i here?9");
    return data_source_id_;
}

BookPrice Symbol::GetReferencePrice() const
{
    return ref_price_;
}

BookPrice Symbol::GetTickSize(BookPrice price, bool up) const
{
    // Note: This function might be pretty slow. Please think twice before using it when production
    // in any latency critical path like strategy signals or alpha calculations.
    BookPrice size{0};

    for (auto it{ticksize_list_.cbegin()}; it != ticksize_list_.cend(); ++it)
    {
        if (up)
        {
            if (price >= it->End)
            {
                continue;
            }
        }
        else
        {
            if (price > it->End)
            {
                continue;
            }
        }

        size = it->Size;
        break;
    }

    return size;
}

ProductType Symbol::GetProductType() const
{
    return product_type_;
}

WarrantType Symbol::GetWarrantType() const
{
    return warrant_type_;
}

int Symbol::GetY() const
{
    return y_;
}

int Symbol::GetM() const
{
    return m_;
}

int Symbol::GetW() const
{
    return w_;
}

const std::string &Symbol::GetCurrency() const
{
    return currency_;
}

const Symbol *Symbol::GetC1Symbol() const
{
    return c1_;
}

const Symbol *Symbol::GetC2Symbol() const
{
    return c2_;
}

bool Symbol::IsRoll() const
{
    return is_roll_;
}

}  // namespace alphaone
