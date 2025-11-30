import pandas as pd
import matplotlib.pyplot as plt

#Load the data that we already got
df = pd.read_csv(
    "data/all_prices.csv",
    header=[0,1],
    index_col=0,
    parse_dates=True
)

#Only use the daily close price for now
close = df["Close"]

#For testing just look at one specific stock
qqq = close["QQQ"]

#Compute the daily returns for that stock
qqq_ret = qqq.pct_change().dropna()

#Equity curve for a buy and sell hold with buycost as original purchase amount in dollars
buy_cost = 1000

equity = buy_cost * (1 + qqq_ret).cumprod()

equity.plot(title="QQQ Buy and Sell Equity Curve")
plt.show()