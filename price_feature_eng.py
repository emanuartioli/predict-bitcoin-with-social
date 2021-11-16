import pandas as pd
import matplotlib.pyplot as plt

price = pd.read_csv('C:/Users/emanu/Desktop/Thesis Project/raw_data/price.csv')
# Date needs to be converted in datetime and then set as index
price['Date'] = pd.to_datetime(price['Date'])
price = price.set_index('Date')
# it's in descending order
price.sort_index(inplace=True)
plt.plot(price.Open)
plt.show()

# to get thresholds we need the percentage daily change
price['change %'] = price['Open'].diff() / price['Open']
price.dropna(inplace=True)

# threshold discovery (TODO: compare predictions for different thresholds)
len(price[price['change %'] > 0.05]) / len(price) # 9.5% chance of gaining more than 5%
len(price[price['change %'] > 0.1]) / len(price) # 2.6% chance
len(price[price['change %'] < -0.05]) / len(price) # 8.1% chance
len(price[price['change %'] < -0.1]) / len(price) # 2.7% chance

# setting threshold
def threshold(daily_change):
    if daily_change > 0.1: return 'high gain'
    elif daily_change > 0.05: return 'gain'
    elif daily_change < -0.1: return 'high loss'
    elif daily_change < -0.05: return 'loss'
    else: return 'stable'
price['threshold'] = price['change %'].apply(threshold)

# daily spread
price['spread %'] = (price['High'] - price['Low']) / price['Low']

# final csv
price.to_csv('C:/Users/emanu/Desktop/Thesis Project/elaboration_data/price_with_features.csv')