import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


#Define the end date (today's date) and the list of tickers we want to pull from
tickers = ['QQQ', 'SPY', 'GLD', 'NQD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
end_date = datetime.today()
print(f"End Date: {end_date}, Tickets: {tickers}")