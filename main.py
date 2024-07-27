from src.utils.compute_utils import ComputeUtils
import pandas as pd
import numpy as np

# beta = Beta("ME","6mo",["SPY", "NDAQ"], [0.5,0.5])
# tickers = ["GOOG","AAPL", "COST","CVS","QLTA","ISCB","ISCG","IWF","IWO","MRK","MSFT","XLE","SFM","LRN","TSM","TTWO","TGNA","HSY","UNH","VBR","VTV","V","ICVT",
#           "JAAA","MINT","SRLN"]

# betas = beta.compute_all_asset_betas(tickers)
# benchmarks = MarketData.get_composite_benchmark_returns(resampling_timeframe="ME", historical_data_timeframe="6mo", composite_benchmark_tickers=["SPY", "NDAQ"], composite_benchmark_weights=[0.5,0.5])

# print(betas)
# print("\n")
# print(benchmarks)

data = {
    "Date": ["Nov 30 2023", "Dec 1 - Dec 31", "Jan 1 - Jan 31", "Feb 1 - Feb 29", "Mar 1 - Mar 31", "Apr 1 - Apr 30", "May 1 - May 31", "Jun 1 - Jun 31"],
    "Total Value": [648023.12, 672957.09, 674563.15, 692541.67, 716338.82, 695328.89, 725236.39, 731479.68]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate returns
df['Return'] = df['Total Value'].pct_change()

# Drop the first NaN value
returns = df['Return'].dropna()

std = ComputeUtils.standard_deviation(returns)
print (std)

print(np.std(returns))