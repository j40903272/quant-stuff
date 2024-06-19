#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include "infrastructure/common/json/Json.hpp"
#include "infrastructure/common/spdlog/spdlog.h"
#include "infrastructure/common/datetime/Timestamp.h"
#include <boost/program_options.hpp>

namespace ao = alphaone;
namespace bp = boost::program_options;

using json = nlohmann::json;

// Callback function to handle the data returned by CURL
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
    } catch(std::bad_alloc &e) {
        // handle memory problem
        return 0;
    }
    return newLength;
}

void fetchURL(const std::string& url, std::string& readBuffer) {
    CURL *curl;
    CURLcode res;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL failed: " << curl_easy_strerror(res) << std::endl;
            return;
        }
    }
}

json fecthMarket() {
    std::string url = "https://fapi.binance.com/fapi/v1/exchangeInfo";
    std::string readBuffer;

    fetchURL(url, readBuffer);

    json jsonData = json::parse(readBuffer);
    /* keys
    assets
    exchangeFilters
    futuresType
    rateLimits
    serverTime
    symbols
    timezone
    */
    // std::cout << jsonData.dump() << std::endl;
    for (auto& [key, val] : jsonData.items())
        std::cout << key << std::endl;
    return jsonData;
}

/*
{"baseAsset":"BTC","baseAssetPrecision":8,"contractType":"PERPETUAL","deliveryDate":4133404800000,"filters":[{"filterType":"PRICE_FILTER","maxPrice":"4529764","minPrice":"556.80","tickSize":"0.10"},{"filterType":"LOT_SIZE","maxQty":"1000","minQty":"0.001","stepSize":"0.001"},{"filterType":"MARKET_LOT_SIZE","maxQty":"120","minQty":"0.001","stepSize":"0.001"},{"filterType":"MAX_NUM_ORDERS","limit":200},{"filterType":"MAX_NUM_ALGO_ORDERS","limit":10},{"filterType":"MIN_NOTIONAL","notional":"100"},{"filterType":"PERCENT_PRICE","multiplierDecimal":"4","multiplierDown":"0.9500","multiplierUp":"1.0500"}],"liquidationFee":"0.012500","maintMarginPercent":"2.5000","marginAsset":"USDT","marketTakeBound":"0.05","maxMoveOrderLimit":10000,"onboardDate":1569398400000,"orderTypes":["LIMIT","MARKET","STOP","STOP_MARKET","TAKE_PROFIT","TAKE_PROFIT_MARKET","TRAILING_STOP_MARKET"],"pair":"BTCUSDT","pricePrecision":2,"quantityPrecision":3,"quoteAsset":"USDT","quotePrecision":8,"requiredMarginPercent":"5.0000","settlePlan":0,"status":"TRADING","symbol":"BTCUSDT","timeInForce":["GTC","IOC","FOK","GTX","GTD"],"triggerProtect":"0.0500","underlyingSubType":["PoW"],"underlyingType":"COIN"}
*/
void fetchKlineData(const std::string& symbol, const std::string& interval, const std::string& filepath) {
    json marketInfo = fecthMarket();
    ao::Timestamp symbol_onboard_date;
    int64_t onboardDate = 0;

    for(auto &symbolInfo : marketInfo["symbols"]) {
        if (symbolInfo["symbol"] == symbol) {
            onboardDate = symbolInfo["onboardDate"].get<int64_t>();
            // symbol_onboard_date = ao::Timestamp::from_epoch_msec(onboardDate);
            SPDLOG_INFO("{}", symbol_onboard_date);
            break;
        }
    }

    std::chrono::milliseconds start_time(onboardDate);

    // Get the current time in milliseconds
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // Loop until the current time reaches the maxEndTime
    while (true) {
        // Check if the current time exceeds the maxEndTime
        if (start_time.count() >= currentTime) {
            std::cout << "Reached the maximum end time. Stopping the program." << std::endl;
            break;
        }

        std::string url = "https://fapi.binance.com/fapi/v1/klines?symbol=" + symbol + "&interval=" + interval + "&startTime=" + start_time.count() + "&limit=" + limit;
        std::string readBuffer;

        fetchURL(url, readBuffer);

        try {
            json jsonData = json::parse(readBuffer);
            std::ofstream csvFile(filepath + "/" + symbol + "_" + interval + ".csv");

            // 2. 自動算column數
            // 3. 從古早時代一路載到現在時間
            // 4. 可以從結尾補充資料
            csvFile << "Open Time,Open,High,Low,Close,Volume,Close Time,Quote Asset Volume,Number of Trades,Taker Buy Base Asset Volume,Taker Buy Quote Asset Volume,Ignored" << "\n";
            for (auto& entry : jsonData) {
                for (auto& value : entry) {
                    csvFile << value << ",";
                }
                csvFile << "\n";
            }


            csvFile.close();
            std::cout << "Data written to CSV file successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "An error occurred: " << e.what() << std::endl;
        }

        // Update the startTime for the next iteration (e.g., by adding a time interval)
        symbol_onboard_date += std::chrono::minutes(1); // Example: Increment by 1 minute

        // Sleep to avoid exceeding the rate limit (adjust the sleep duration as needed)
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Sleep for 1 second
    }
    
}

int main(int argc, char **argv)
{
    // parse arguments
    // clang-format off
    bp::options_description descriptions("GetBinanceData");
    descriptions.add_options()
        ("symbol,s", bp::value<std::string>()->default_value("BTCUSDT"), "symbol")
        ("interval,t", bp::value<std::string>()->default_value("1h"), "interval")
        ("filepath,p", bp::value<std::string>()->required(), "filepath")
        ("help,h", "help");
    // clang-format on
    boost::program_options::variables_map vm;
    try
    {
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, descriptions), vm);
        if (vm.count("help"))
        {
            std::cout << descriptions << '\n';
            exit(1);
        }
        boost::program_options::notify(vm);
    }
    catch (const boost::program_options::error &e)
    {
        std::cout << "ERROR: " << e.what() << '\n';
        std::cout << descriptions << '\n';
        exit(1);
    }
    std::string symbol = vm["symbol"].as<std::string>();
    std::string interval = vm["interval"].as<std::string>();
    std::string filepath = vm["filepath"].as<std::string>();

    // fecthMarket();
    fetchKlineData(symbol, interval, filepath);

    return 0;
}
