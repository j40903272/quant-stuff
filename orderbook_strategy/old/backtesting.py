import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_top_percent_values(dataframe, column_name, quantile_value):
    # Filter non-zero values
    non_zero_values = dataframe[dataframe[column_name] != 0]

    # Calculate the threshold for the specified quantile
    threshold = non_zero_values[column_name].quantile(quantile_value)

    # Filter values above the threshold
    top_percent_values = dataframe[dataframe[column_name] > threshold]

    # Sort the DataFrame based on 'bid_change_percentage' in descending order
    top_percent_values = top_percent_values.sort_values(column_name, ascending=False)

    return top_percent_values

binance = pd.read_csv('./aligned_binance.csv',engine="pyarrow")
binance.set_index('timestamp', inplace=True)

okx = pd.read_csv('./aligned_okx.csv',engine="pyarrow")
okx.set_index('timestamp', inplace=True)

binance['bid_change_percentage'] = ((binance['bid'] - binance['bid'].shift(1)) / binance['bid'].shift(1))
okx['bid_change_percentage'] = ((okx['bid'] - okx['bid'].shift(1)) / okx['bid'].shift(1))

# Filter non-zero values in 'bid_change_percentage'
binance_bid = get_top_percent_values(binance, 'bid_change_percentage', 0.999)
okx_bid = get_top_percent_values(okx, 'bid_change_percentage', 0.99)

entry = binance_bid.index.tolist()

result = []

for transation_latency in range(20, 100):
    for offset in range(10, 50):
        sum = 0
        for i in entry:
            open = i + transation_latency 
            close = open + offset
            sum += okx.loc[close]['bid'] - okx.loc[open]['ask']

        result.append({
            'transation_latency':transation_latency,
            'offset':offset,
            'sum':sum * 100 - 0.10
        })

sorted_result = sorted(result, key=lambda x: x['sum'], reverse=True)

# # Print the sorted result
# for item in sorted_result[:100]:
#     print(item)

# # Set up subplots
# plt.figure(figsize=(12, 5))

# # Plot histogram for 'binance_bid'
# plt.subplot(1, 2, 1)
# plt.hist(binance_bid['bid_change_percentage'], bins=np.arange(min(binance_bid['bid_change_percentage']), max(binance_bid['bid_change_percentage'])+0.0001, step=0.0001), edgecolor='black')
# plt.title('Binance - Distribution of Bid Change Percentage')
# plt.xlabel('Bid Change Percentage')
# plt.ylabel('Frequency')

# # Plot histogram for 'okx_bid'
# plt.subplot(1, 2, 2)
# plt.hist(okx_bid['bid_change_percentage'], bins=np.arange(min(okx_bid['bid_change_percentage']), max(okx_bid['bid_change_percentage'])+0.0001, step=0.0001), edgecolor='black')
# plt.title('OKX - Distribution of Bid Change Percentage')
# plt.xlabel('Bid Change Percentage')
# plt.ylabel('Frequency')

# # Adjust layout
# plt.tight_layout()
# plt.show()

# Extract transation_latency and sum from the list of dictionaries
transation_latency_values = [item['transation_latency'] for item in sorted_result]
sum_values = [item['sum'] for item in sorted_result]

# Plot the data
plt.scatter(transation_latency_values, sum_values,  color='blue')
plt.title('Transaction Latency vs. Sum')
plt.xticks(np.arange(20, 100, 1))  
plt.yticks(np.arange(0, 2, 0.05)) 
plt.xlabel('Transaction Latency')
plt.ylabel('Sum')
plt.grid(True)
plt.show()



