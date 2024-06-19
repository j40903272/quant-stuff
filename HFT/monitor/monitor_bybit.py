#%%
import os
import json
import numpy as np
# import discord
# from discord.ext import tasks, commands
import nest_asyncio
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import time
from minute_plotter import PLOTSAVER
#%%
def plot_dataframe(info : dict, header_columns=0, header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',):
    df = pd.DataFrame(info).T
    df.columns = ['direction','chance','maker/taker','maker/maker','taker/taker','max spread','<0.001']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    mpl_table = ax.table(cellText=df.values,
             colLabels=df.columns,
             cellLoc='center',
             loc='center')
    
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    
    return ax.get_figure(), ax

def plot_trade_records(data, header_columns=0, header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w'):
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    mpl_table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center')
    
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        
    return ax.get_figure(), ax



def get_arbitrage_info(output_dict):    

    info = {}
    txt = '【Bybit Bitget Arbitrage Monitor】\n'

    for key in output_dict.keys():
        data_dict = {}
        chance = len(output_dict[key])
        if key == 'operation_1': 
            txt += f'--L(bitget)S(bybit) total chances = {chance} in an hour--\n' 
            if chance == 0:
                info['operation_1'] = ['L(bitget)S(bybit)',0,0,0,0,0,len(output_dict['convergence'])]
            else:
                for item in output_dict[key]:
                    k = list(item.keys())[0]
                    values = list(item.values())[0]
                    data_dict[k] = values

                df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['maker/taker', 'maker/maker','taker/taker'])
                df.index.name = 'Ts'

                info['operation_1'] = ['L(bitget)S(bybit)',len(df),round(df['maker/taker'].mean(),5),round(df['maker/maker'].mean(),5),round(df['taker/taker'].mean(),5),
                                            max(df['maker/taker']),len(output_dict['convergence'])]
        elif key == 'operation_2': 
            txt += f'--L(bybit)S(bitget) total chances = {chance} in an hour--\n' 
            if chance == 0:
                info['operation_2'] = ['L(bybit)S(bitget)',0,0,0,0,0,len(output_dict['convergence'])]
            else:
                for item in output_dict[key]:
                    k = list(item.keys())[0]
                    values = list(item.values())[0]
                    data_dict[k] = values

                df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['maker/taker', 'maker/maker','taker/taker'])
                df.index.name = 'Ts'

                info['operation_2'] = ['L(bybit)S(bitget)',len(df),round(df['maker/taker'].mean(),5),round(df['maker/maker'].mean(),5),round(df['taker/taker'].mean(),5),
                                            max(df['maker/taker']),len(output_dict['convergence'])]

    if info['operation_1'] or info['operation_2'] != []:
        fig, ax = plot_dataframe(info)
        txt += 'bybit_bitget\n'
        fig.savefig('./data/bybit_bitget_arb.png',bbox_inches='tight')
        time.sleep(1)


    return txt

def msg_split_list(msg):
    msg_list = msg.split('\n')
    msg_list = [item for item in msg_list if item != '']
    return msg_list

def get_latest_file(folder_path):
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
    sorted_files = sorted(file_paths, key=os.path.getmtime, reverse=True)
    if sorted_files:
        return sorted_files[0]
    else:
        return None
    
class DISCORD():
    def __init__(self, file_name = 'research/'):

        nest_asyncio.apply()

        # client = discord.Client(intents=discord.Intents.default())

        # Alex
        # server_id = 1099688214513582190
        # channel_id = 1147913390216458270
        # WQZnRE1gZzAmAOosR7z5uGQ6H57j8cO7FGS2_DHu5EOjaqwsKr_gzwc39RTyU0mY1QK0

        # server_id = 943767327642640404
        # token = 'OTQzNzcxNzgyODAyOTkzMTUz.Yg359w.dhUF3K8J5ohirF3P9N3qYoRxoEU'
        # channel_id = 1127532652132585482

        # server_id = 957772998125957180
        # token = 'MTEyNzIyMDY0MzA1ODk1ODM3OA.G28RT6.pJL3L0bAKTd6V1Vj4gbJmbnjt_fC3k6qALE6Ms'
        # channel_id = 1127221680842997851

        # @client.event
        # # 當機器人完成啟動
        # async def on_ready():
        #     print(f"目前登入身份 --> {client.user}")
        #     Send_txt.start()

        # @tasks.loop(minutes=1)
        async def Send_txt():
            
        #     channel = client.get_guild(server_id).get_channel(channel_id)

            if(datetime.datetime.now().minute in [1]):
    
        #         channel = client.get_guild(server_id).get_channel(channel_id)

                file_path = get_latest_file(file_name)
                # print(file_path)
                with open(file_path, 'r') as file:
                    output_dict = json.load(file)

                try:
                    msg = get_arbitrage_info(output_dict)
                except Exception as e:
                    msg = str(e) + '/n'

                txts = msg_split_list(msg)
                print(txts)
                # for txt in txts:
                #     if txt == 'bybit_bitget':
                #         await channel.send(file=discord.File('./data/bybit_bitget_arb.png'))
                #     else:
                #         await channel.send(txt)

                # print(datetime.datetime.now(), channel)
            
            if(datetime.datetime.now().minute in [2]):
                file_path = get_latest_file('./trade/')
                with open(file_path, 'r') as file:
                    trade_records = json.load(file)
                print(file_path)
                fig, ax = plot_trade_records(trade_records)
                fig.savefig('./data/trading_records_df.png',bbox_inches='tight')
                time.sleep(10)
                
                # await channel.send(file=discord.File('./data/trading_records_df.png'))
            
                if(datetime.datetime.now().hour in [0]):
                    plotter = PLOTSAVER()
                    plotter.save_plot()
                    # await channel.send('Daily 1m Spread Overview')
                    # await channel.send(file=discord.File('./data/daily_spread_1m.png'))
                    
        # try:
        #     client.run(token)
        # except:
        #     pass

discord = DISCORD()

# #%%
# file_path = get_latest_file('./research/trade/')
# with open(file_path, 'r') as file:
#     trade_records = json.load(file)
# fig, ax = plot_trade_records(trade_records)
# fig.show()
# # fig.savefig('./data/trading_records_df.png',bbox_inches='tight')
