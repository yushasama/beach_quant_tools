from ..utils.compute_utils import ComputeUtils
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np

class MarketDataYFinance:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def validate_ticker(ticker: str) -> None:
        """
        Checks if the given asset ticker is valid by attempting to download historical data.
        Raises an exception if the ticker is not valid.

        Args:
            ticker (str): The asset ticker symbol.

        Raises:
            ValueError: If the ticker is not valid.
        """
        asset_data = yf.download(ticker, period='1d')

        if asset_data.empty:
            errorMessage = f"{ticker} is not valid, please provide a valid ticker."
            raise ValueError(errorMessage)
        
        if ticker == "FTX":
            raise ValueError(f"Good one xD. {ticker} is not valid, please provide a valid ticker.")

    @staticmethod
    def get_asset_returns(ticker: str, resampling_timeframe: str, historical_data_timeframe: str) -> List[float]:
        """
        Gets asset returns for the given ticker and timeframe.

        Parameters:
            ticker (str): The asset ticker symbol.
            resampling_timeframe (str): The timeframe for resampling ('D' for daily, 'W' for weekly, 'ME' for monthly).
            historical_data_timeframe (str): The historical data timeframe (e.g., '1y' for one year, '6mo' for six months).

        Returns:
            List[float]: List of {resampling_timeframe} asset returns rounded to 2 decimal places.
        """

        # Validate the ticker
        MarketDataYFinance.validate_ticker(ticker)

        # Download historical asset data
        asset_data = yf.download(ticker, period=historical_data_timeframe)

        # Ensure the index is a DatetimeIndex
        asset_data.index = pd.to_datetime(asset_data.index)

        # Resample the data based on the timeframe
        valid_timeframes = ['D', 'W', 'ME', 'Q', 'A']

        if resampling_timeframe not in valid_timeframes:
            raise ValueError(f"Invalid resampling_timeframe. Use one of {valid_timeframes}.")

        # Resample and calculate returns
        asset_resampled = asset_data['Adj Close'].resample(resampling_timeframe).ffill()
        asset_returns = asset_resampled.pct_change().dropna()

        
        # Creates a list through list comprehension and round returns from asset_returns to two decimal places
        return [round(return_val, 2) for return_val in asset_returns]
    
    @staticmethod
    def get_composite_benchmark_returns(composite_benchmark_tickers: list[str], composite_benchmark_weights: float, resampling_timeframe: str, historical_data_timeframe: str) -> List[float]:
        """
        Fetches returns for the composite benchmark comprised of individual assets and their respective weights in the benchmark
        
        Parameters:
            tickers (list[str]): The list of asset ticker symbol chosen for the composite benchmark.
            resampling_timeframe (str): The timeframe for resampling ('D' for daily, 'W' for weekly, 'ME' for monthly).
            historical_data_timeframe (str): The historical data timeframe (e.g., '1y' for one year, '6mo' for six months).

        Returns:
            List[float] of composite weighted average {resampling_timeframe} benchmark returns rounded to 2 decimal places.

        Formula (laTeX):
        \begin{align*}
        n &= \text{number of resampled timeframes within the historical period} \\
        m &= \text{number of benchmark tickers} \\
        \psi_{i,j} &= \text{the return of the } j\text{-th benchmark asset for the } i\text{-th resampled timeframe} \\
        R_b &= \{r_{b_i}\}_{i=1}^{n} \\
        r_{b_i} &= \Psi_i \cdot w, \text{ a float representing the weighted average return of the } i\text{-th resampled timeframe, given by the dot product} \\
        \Psi &= \{ \psi_{i,j} \}_{i=1,j=1}^{n,m}, \text{ an } n \times m \text{ matrix of returns for each benchmark ticker and resampled timeframe} \\
        \Psi_i &= \text{the } i\text{-th row vector of the matrix } \Psi, \text{ representing the returns for each benchmark asset for the } i\text{-th resampled timeframe} \\
        r_i &= \text{vector of } \{ \psi_{i,j} \}_{j=1}^{m}, \text{ with } i \text{ representing the current } i\text{-th resampled timeframe and } j \text{ being the benchmark ticker} \\
        w &= \text{vector of normalized benchmark weights}
        \end{align*}
        """

        if len(composite_benchmark_tickers) != len(composite_benchmark_weights):
            raise ValueError("'composite_benchmark_tickers' and 'composite_benchmark_weights' must have equal input size.")
        
        # Normalize composite benchmark weights
        weights = ComputeUtils.normalize_weights(composite_benchmark_weights)
        
        returns_matrix = []

        for ticker in composite_benchmark_tickers:
            psi_j = MarketDataYFinance.get_asset_returns(ticker, resampling_timeframe, historical_data_timeframe)
            returns_matrix.append(psi_j)

        # Convert the list of lists into a DataFrame and transpose it, assigning dates and returns to rows and columns respectively.
        df = pd.DataFrame(returns_matrix).T
        df.columns = composite_benchmark_tickers

        # Assign dates to columns
        df.index = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq=resampling_timeframe)

        # Convert the DataFrame to a Numpy matrix
        returns_matrix = df.values
        weights_vector = np.array(composite_benchmark_weights)

        # Compute weighted returns of the composite benchmark using matrix multiplication.
        weighted_returns = np.dot(returns_matrix, weights_vector)

        # Creates a list through list comprehension and round returns from weighted_returns to two decimal places
        return [round(weighted_return_val, 2) for weighted_return_val in weighted_returns]

    @staticmethod
    def get_asset_price(ticker: str) -> float:
        """
        Fetches the current price of the given asset ticker.

        Args:
            ticker (str): The ticker symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            float: The current asset price rounded to 2 decimal places.
        """
        # Validate the ticker
        MarketDataYFinance.validate_ticker(ticker)

        asset = yf.Ticker(ticker)
        current_price = asset.history(period='1d')['Close'][0]

        return round(current_price, 2)

    @staticmethod
    def get_asset_std_deviation(ticker: str, historical_data_timeframe: str) -> float:
        """
        Computes the standard deviation of the given asset ticker's returns.

        Args:
            ticker (str): The asset ticker symbol (e.g., 'AAPL' for Apple Inc.).
            historical_data_timeframe (str): The historical data timeframe (e.g., '1y' for one year, '6mo' for six months).

        Returns:
            float: The standard deviation of the asset's returns.
        """

        # Validate the ticker
        MarketDataYFinance.validate_ticker(ticker)

        # Download historical asset data
        asset_data = yf.download(ticker, period=historical_data_timeframe)

        # Calculate daily returns
        asset_data['Returns'] = asset_data['Adj Close'].pct_change()

        # Calculate the standard deviation of returns
        std_dev = asset_data['Returns'].std()

        return round(std_dev, 2)