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

def strat_buy_and_hold(prices, initial_capital=1.0):
    #Invest all capital at t=0 and hold till the end
    return initial_capital * (prices[-1] / prices[0])

def strategy_drop_5pct_hold_2m(prices, drop_threshold=-0.05, hold_days=42, initial_capital=1.0):
    prices = pd.Series(prices).reset_index(drop=True)
    returns = prices.pct_change()

    equity = initial_capital
    i = 1  # start at 1 because returns[0] is NaN

    while i < len(prices) - 1:
        # Check for drop on day i (compared to previous close)
        if returns.iloc[i] <= drop_threshold:
            entry_idx = i + 1
            if entry_idx >= len(prices):
                break

            entry_price = prices.iloc[entry_idx]

            exit_idx = min(entry_idx + hold_days, len(prices) - 1)
            exit_price = prices.iloc[exit_idx]

            trade_ret = exit_price / entry_price - 1
            equity *= (1 + trade_ret)

            # Skip ahead so trades don't overlap
            i = exit_idx + 1
        else:
            i += 1

    return equity

#Simulate MC function
def sim_paths(S0, mu, sigma, time=252, n_sims=10000):
    dt = 1
    rand_norm = np.random.normal(0, 1, (time, n_sims))
    sim_log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_norm
    log_price_relatives = sim_log_rets.cumsum(axis=0)
    price_paths = S0 * np.exp(log_price_relatives)
    return price_paths