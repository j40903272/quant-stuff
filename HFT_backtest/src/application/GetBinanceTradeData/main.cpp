#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include "infrastructure/common/json/Json.hpp"

#include <boost/program_options.hpp>

namespace bp = boost::program_options;

using json = nlohmann::json;

size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

json GetAggregatedTrades(const std::string& symbol, const std::string& startTime, const std::string& endTime, int limit) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    std::string url = "https://fapi.binance.com/fapi/v1/aggTrades?symbol=" + symbol + "&startTime=" + startTime + "&endTime=" + endTime + "&limit=" + std::to_string(limit);

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL failed: " << curl_easy_strerror(res) << std::endl;
            return json();
        }
    }

    json jsonResponse = json::parse(readBuffer);
    return jsonResponse;
}

int main(int argc, char **argv)
{
    // parse arguments
    // clang-format off
    bp::options_description descriptions("GetBinanceData");
    descriptions.add_options()
        ("symbol,s", bp::value<std::string>()->default_value("BTCUSDT"), "symbol")
        ("startTime,t", bp::value<std::string>()->default_value("1609459200000"), "startTime")
        ("endTime,e", bp::value<std::string>()->default_value("1609545600000"), "endTime")
        ("filepath,p", bp::value<std::string>()->default_value("./2020-05-26/trade.txt"), "filepath")
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
    std::string startTime = vm["startTime"].as<std::string>();
    std::string endTime = vm["endTime"].as<std::string>();
    std::string filepath = vm["filepath"].as<std::string>();
    int limit = 1000;

    json trades = GetAggregatedTrades(symbol, startTime, endTime, limit);
    std::cout << trades.dump(4) << std::endl;

    // Write JSON to a text file
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << trades.dump(4);
        file.close();
        std::cout << "Data written to trades_data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

    return 0;
}
