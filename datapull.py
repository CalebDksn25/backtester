import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


#Define the end date (today's date) and the list of tickers we want to pull from
tickers = ['QQQ', 'SPY', 'GLD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
end_date = datetime.today()

#Set the start date
start_date = end_date - timedelta(days=5*365)

df = yf.download(tickers=tickers, start=start_date, end=end_date)
df.to_csv("data/all_prices.csv")
#print(df.columns)

#Calculate the change in price per day
price_dif = (df["Close"] - df["Open"]) / df["Open"]
price_dif = (price_dif * 100).round(5)
price_dif.to_csv("data/price_dif_pct.csv")

#Plot the data and show visualization
price_dif.plot()
df["Open"].plot()
plt.show()