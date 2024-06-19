# backtest-system

[![Project-Guidelines](https://img.shields.io/badge/Project-Guidelines-blue.svg)](https://github.com/kevinyin9/Alex_backtest_system/blob/add-rule-md/RULE.md)

### package setup 
```bash
pip install quantstats
pip install vectorbt
pip install numpy 
pip install pandas
pip install datetime
pip install hiplot
pip install statsmodels
pip install Cython
pip install ciso8601
pip install optuna
pip install hyperopt
pip install ccxt
pip install -U kaleido
pip install TA-Lib
```

### For Windows User
Download [ta-lib-0.4.0-msvc.zip](https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-msvc.zip/download?use_mirror=nchc) and unzip to C:\ta-lib.

This is a 32-bit binary release. If you want to use 64-bit Python, you will need to build a 64-bit version of the library. Some unofficial (and unsupported) instructions for building on 64-bit Windows 10, here for reference:

1. Download and Unzip ta-lib-0.4.0-msvc.zip
1. Move the Unzipped Folder ta-lib to C:\
1. Download and Install Visual Studio Community (2015 or later)
    * Remember to Select [Visual C++] Feature
1. Build TA-Lib Library
    1. From Windows Start Menu, Start [VS2015 x64 Native Tools Command Prompt]
    1. Move to C:\ta-lib\c\make\cdr\win32\msvc
    1. Build the Library **nmake**


### core code init (First time user)
```bash
cd backtest_system
python3 backtest_engine/setup_core.py build_ext
```

## Load Crypto Price Data
```bash
cd ./backtest_system
python3 ./data_loader/binance/binance_data_download.py -s BTCUSDT -t 1m --type SPOT
python3 ./data_loader/binance/binance_data_download.py -s ETHUSDT -t 1m --type SPOT
python3 ./data_loader/binance/binance_data_download.py -s BTCUSDT -t 1m --type UPERP
python3 ./data_loader/binance/binance_data_download.py -s ETHUSDT -t 1m --type UPERP
```

## Strategy File Location
- Strategy: ./backtest_system/backtest_engine/strategies/
- StrategyParameters: ./backtest_system/backtest_engine/strategies_parameters/
- After Creating New Strategy: ./backtest_engine/strategy_list.py add your strategy to the import

## Run Backtest
``` bash
python3 ./backtest_engine/backtest_initiate/backtest_init.py --strategy_logic ForceStrategy --strategy_config ForceStrategy --start "2023-04-01 18:30:00" --end "2023-06-13 18:30:00"
python3 ./backtest_engine/backtest_initiate/backtest_init.py --strategy_logic ForceStrategy --strategy_config ForceStrategy --random_std 0.001
```

## Tuning Hyperparameter
``` bash
python3 ./backtest_engine/strategy_tuner/random_tuner.py --strategy_logic SMAStrategy --symbol ETH --tuning_num 50 --tuning_json 00_range_tuner
```


- Setting exchange tx fee, initial capital, and leverage multiplier
- location: ./backtest_system/backtest_engine/exchange_enum.py

