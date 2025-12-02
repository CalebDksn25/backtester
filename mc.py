import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/average_price.csv")

#Make sure the date is a date and sorted
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

#Pick a ticker to simulate
prices = df["QQQ"]

log_returns = np.log(prices / prices.shift(1)).dropna()

#Estimate the daily mean and volatility from history
mu = log_returns.mean() #Daily log return
sigma = log_returns.std() #Daily volatility

#Set simulation parameters
time = 252 #Number of days to sim
n_sims = 10000 #Number of sims to run
S0 = prices.iloc[-1] #Starting price is last price in data

#Run the simulation
dt = 1

#Random shocks
rand_norm = np.random.normal(0, 1, (time, n_sims))

#Simulate log returns
sim_log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_norm

#Cumulative3 log returns -> relative price changes
log_price_relatives = sim_log_rets.cumsum(axis=0)

#Sim price path
price_paths = S0 * np.exp(log_price_relatives)

#Look at the outcome
final_prices = price_paths[-1, :]

expected_final = final_prices.mean()
p05 = np.percentile(final_prices, 5) #5th percentile (Bad outcome)
p50 = np.percentile(final_prices, 50) #Median outcome
p95 = np.percentile(final_prices, 95) #95th percentile (Good outcome)

print("Start price: ", S0)
print("Expected Final: ", expected_final)
print("5th Percentile Final: ", p05)
print("50th Percentile Final: ", p50)
print("95th Percentile Final: ", p95)

#Plot some of the simulated paths
plt.figure()
for i in range(100): #Show 100 random paths
    plt.plot(price_paths[:, i], alpha=0.3)
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Simulated MC Price Paths for QQQ")
plt.show()

#Histogram of final prices
plt.figure()
plt.hist(final_prices, bins=50)
plt.xlabel("Final Price")
plt.ylabel("Frequency")
plt.title("Distribution of final prices after 252 days".format(time))
plt.show()