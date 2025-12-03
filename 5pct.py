import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. LOAD HISTORICAL DATA
# ----------------------------------------------------------

df = pd.read_csv("data/average_price.csv")

# Make sure the date is parsed and sorted
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# Pick the ticker column to simulate (QQQ in your example)
prices = df["QQQ"]

# ----------------------------------------------------------
# 2. ESTIMATE PARAMETERS FROM HISTORY
# ----------------------------------------------------------

log_returns = np.log(prices / prices.shift(1)).dropna()

mu = log_returns.mean()        # daily log mean
sigma = log_returns.std()      # daily volatility

# ----------------------------------------------------------
# 3. MONTE CARLO SIMULATOR
# ----------------------------------------------------------

def simulate_paths(S0, mu, sigma, time=252, n_sims=10000):
    dt = 1
    rand_norm = np.random.normal(0, 1, (time, n_sims))
    sim_log_rets = (mu - 0.5 * sigma**2) * dt + sigma * rand_norm
    log_price_relatives = sim_log_rets.cumsum(axis=0)
    price_paths = S0 * np.exp(log_price_relatives)
    return price_paths  # shape [time, n_sims]

# Simulation parameters
TIME = 252
N_SIMS = 10000
S0 = prices.iloc[-1]

price_paths = simulate_paths(S0, mu, sigma, time=TIME, n_sims=N_SIMS)

# ----------------------------------------------------------
# 4. STRATEGY IMPLEMENTATION
# ----------------------------------------------------------

def strategy_buy_and_hold(prices, initial_capital=1.0):
    return initial_capital * (prices[-1] / prices[0])

def strategy_drop_5pct_hold_42_days(prices, drop_threshold=-0.05, hold_days=42, initial_capital=1.0):
    # Convert prices to pandas Series for pct_change
    prices = pd.Series(prices).reset_index(drop=True)
    returns = prices.pct_change()

    equity = initial_capital
    i = 1  # start at day 1 (day 0 has no return)

    while i < len(prices) - 1:
        # Check if today's drop is >= 5%
        if returns.iloc[i] <= drop_threshold:

            entry_idx = i + 1
            if entry_idx >= len(prices):
                break

            entry_price = prices.iloc[entry_idx]

            exit_idx = min(entry_idx + hold_days, len(prices) - 1)
            exit_price = prices.iloc[exit_idx]

            trade_return = exit_price / entry_price - 1
            equity *= (1 + trade_return)

            # Skip forward so trades don't overlap
            i = exit_idx + 1

        else:
            i += 1

    return equity

# ----------------------------------------------------------
# 5. RUN STRATEGY ON ALL SIMULATED PATHS
# ----------------------------------------------------------

bh_equity = np.zeros(N_SIMS)
drop_equity = np.zeros(N_SIMS)

for j in range(N_SIMS):
    path = price_paths[:, j]
    bh_equity[j] = strategy_buy_and_hold(path)
    drop_equity[j] = strategy_drop_5pct_hold_42_days(path)

bh_returns = bh_equity - 1
drop_returns = drop_equity - 1

# ----------------------------------------------------------
# 6. PRINT SUMMARY STATISTICS
# ----------------------------------------------------------

print("--------------------------------------------------")
print("Monte Carlo Backtest Results")
print("--------------------------------------------------")

print("BUY & HOLD:")
print("  Mean return:", np.mean(bh_returns))
print("  5th pct:", np.percentile(bh_returns, 5))
print("  50th pct:", np.percentile(bh_returns, 50))
print("  95th pct:", np.percentile(bh_returns, 95))
print()

print("DROP -5% STRATEGY:")
print("  Mean return:", np.mean(drop_returns))
print("  5th pct:", np.percentile(drop_returns, 5))
print("  50th pct:", np.percentile(drop_returns, 50))
print("  95th pct:", np.percentile(drop_returns, 95))
print()

# ----------------------------------------------------------
# 7. VISUALIZATIONS
# ----------------------------------------------------------

# Price paths
plt.figure(figsize=(10,5))
for i in range(100):
    plt.plot(price_paths[:, i], alpha=0.25)
plt.title("Monte Carlo Simulated Price Paths")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Return distributions
plt.figure(figsize=(10,5))
plt.hist(bh_returns, bins=50, alpha=0.5, label="Buy & Hold")
plt.hist(drop_returns, bins=50, alpha=0.5, label="Drop -5% Strategy")
plt.xlabel("Final Return")
plt.ylabel("Frequency")
plt.title("Distribution of Final Returns (MC Simulation)")
plt.legend()
plt.show()

print("Done.")