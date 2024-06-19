import os
import pandas as pd

def vbt_saver_printer(
    user_home_path,
    save_dir, 
    strategy_logic, 
    strategy_config,
    stats_df,
    position_df,
    bt_df,
    pf,
    asset_return,
    asset_value,
    cumulative_returns,
    designer,
    design_date,
    status
    ):

    save_dir = f'{user_home_path}/backtest_system/data_center/vector_bt'
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(save_dir + f"/{strategy_logic}/stats/{strategy_config}", exist_ok=True)
    os.makedirs(save_dir + f"/{strategy_logic}/position/{strategy_config}", exist_ok=True)
    os.makedirs(save_dir + f"/{strategy_logic}/metric_df/{strategy_config}", exist_ok=True)
    os.makedirs(save_dir + f"/{strategy_logic}/asset/{strategy_config}", exist_ok=True)
    os.makedirs(save_dir + f"/{strategy_logic}/trade_pnl/{strategy_config}", exist_ok=True)
    # saving
    stats_df.to_csv(save_dir + f"/{strategy_logic}/stats" + f"/{strategy_config}/{strategy_config}_stats_vbt.csv")
    position_df.to_csv(save_dir + f"/{strategy_logic}/position"+ f"/{strategy_config}/{strategy_config}_position_vbt.csv")
    bt_df.to_csv(save_dir + f"/{strategy_logic}/metric_df" + f"/{strategy_config}/{strategy_config}_metric_df_vbt.csv")
    trade_record = pf.positions.records_readable
    trade_record.to_csv(save_dir + f"/{strategy_logic}/trade_pnl" + f"/{strategy_config}/{strategy_config}_trade_pnl_vbt.csv")
    asset_return.to_csv(save_dir + f"/{strategy_logic}/asset" + f"/{strategy_config}/{strategy_config}_asset_return_vbt.csv")
    asset_value.to_csv(save_dir + f"/{strategy_logic}/asset" + f"/{strategy_config}/{strategy_config}_asset_value_vbt.csv")
    cumulative_returns.to_csv(save_dir + f"/{strategy_logic}/asset" + f"/{strategy_config}/{strategy_config}_cumulative_returns_vbt.csv")
    print("VBT Success!")
    # credit
    print('---------')
    print(f"The strategy designer is {designer}")
    print(f'Designed in {design_date}')
    print('---------')
    print(f"Strategy Type: {status}")
    print('---------')