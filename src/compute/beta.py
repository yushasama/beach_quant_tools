from ..utils.compute_utils import ComputeUtils
from ..market_data.market_data_yfinance import MarketDataYFinance
from typing import List, Dict
import numpy as np

class Beta:
    """
    A class to compute the beta of a asset or a portfolio relative to a market benchmark.

    Attributes:
        market_return_benchmark_ticker (str): The respective ticker of the market benchmark used for beta calculation ('SPY', 'NDAQ', or 'DJI').

    Methods:
        compute_all_asset_betas(tickers: List[str]) -> Dict[str, float]: Computes the beta of a list of assets.
        compute_portfolio_beta(tickers: List[str], portfolio_weights: List[float]) -> float: Computes the beta of a portfolio.
    """

    def __init__(self, resampling_timeframe: str, historical_data_timeframe: str, market_return_benchmark_tickers: str, market_return_benchmark_ticker_weights = {str,float}) -> None:
        """
        Constructs all the necessary attributes for the Beta object.

        Parameters:
            resampling_timeframe (str): The timeframe for resampling ('D', 'W', 'M', 'Q', 'A').
            historical_data_timeframe (str): The historical data timeframe (e.g., '1y' for one year, '6mo' for six months).
            market_return_benchmark_ticker (str): The respective ticker of the market benchmark used for beta calculation ('SPY', 'NDAQ', or 'DJI').
        """

        self.resampling_timeframe = resampling_timeframe
        self.historical_data_timeframe = historical_data_timeframe
        self.market_return_benchmark_tickers = market_return_benchmark_tickers
        self.market_return_benchmark_ticker_weights = market_return_benchmark_ticker_weights

    @staticmethod
    def _compute_asset_beta(asset_returns: List[float], market_returns: List[float]) -> float:
        """
        Helper function to compute the beta of a asset.

        Parameters:
            asset_returns (List[float]): List of asset returns.
            market_returns (List[float]): List of market returns.

        Returns:
            float: The beta of the asset.
        """

        if len(asset_returns) != len(market_returns):
            raise ValueError("The input sizes of 'asset_returns' and 'market_returns' must be equal.")

        covariance_matrix = np.cov(asset_returns, market_returns)
        covariance = covariance_matrix[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance

        return round(beta, 2)

    @staticmethod
    def _compute_portfolio_beta(portfolio_weights: List[float], asset_betas: List[float]) -> float:
        """
        Helper function to compute the beta of a portfolio.

        Parameters:
            portfolio_weights (List[float]): List of portfolio weights.
            asset_betas (List[float]): List of asset betas.

        Returns:
            float: The beta of the portfolio.
        """

        if len(portfolio_weights) != len(asset_betas):
            raise ValueError("The input sizes of 'portfolio_weights' and 'asset_betas' must be equal.")

        portfolio_beta = np.dot(portfolio_weights, asset_betas)
        return portfolio_beta

    def compute_all_asset_betas(self, tickers: List[str]) -> Dict[str, float]:
        """
        Computes the beta of all given assets.

        Parameters:
            tickers (List[str]): List of asset tickers. Example: tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NFLX', 'BRK-B', 'JPM', 'V']

        Returns:
            Dict[str, float]: A dictionary mapping asset tickers to their respective beta values.
        """

        asset_betas = {}

        benchmark_market_returns = MarketDataYFinance.get_composite_benchmark_returns(
            self.market_return_benchmark_tickers,
            self.market_return_benchmark_ticker_weights,
            self.resampling_timeframe,
            self.historical_data_timeframe
        )

        for ticker in tickers:

            # Checks if inputted ticker is valid
            MarketDataYFinance.validate_ticker(ticker)

            indv_asset_returns = MarketDataYFinance.get_asset_returns(ticker, self.resampling_timeframe, self.historical_data_timeframe)
            beta = self._compute_asset_beta(indv_asset_returns, benchmark_market_returns)
            asset_betas[ticker] = beta

        return asset_betas

    def compute_portfolio_beta(self, tickers: List[str], portfolio_weights: List[float]) -> float:
        """
        Computes the beta of a portfolio.

        Parameters:
            tickers (List[str]): List of asset tickers. Example: tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NFLX', 'BRK-B', 'JPM', 'V']
            portfolio_weights (List[float]): List of portfolio weights corresponding to the assets from 'tickers' parameter. Weights should sum up to 1.0.

        Returns:
            float: The beta of the portfolio.
        """

        # Normalize the inputted portfolio weights, ensuring they all add up to 1.0.
        portfolio_weights = ComputeUtils.normalize_weights(portfolio_weights)

        asset_betas = list(self.compute_all_asset_betas(tickers).values())

        return self._compute_portfolio_beta(portfolio_weights, asset_betas)