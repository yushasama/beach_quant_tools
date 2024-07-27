import numpy as np
from typing import List

def compute_sharpe(portfolio_returns: List[float], risk_free_rate: float) -> float:
    """
    Computes the Sharpe ratio for a given portfolio.

    Args:
        portfolio_returns (List[float]): A list of portfolio returns.
        risk_free_rate (float): The risk-free rate.

    Returns:
        float: The Sharpe ratio of the portfolio. Rounded to 2 decimal places.
    """
    # Convert the list to a numpy array for numerical operations
    portfolio_returns = np.array(portfolio_returns)

    # Calculate the average return of the portfolio
    average_return = np.mean(portfolio_returns)

    # Calculate the standard deviation of the portfolio returns
    std_deviation = np.std(portfolio_returns)

    # Calculate the Sharpe ratio
    sharpe_ratio = (average_return - risk_free_rate) / std_deviation

    # Round the result if required
    return round(sharpe_ratio, 2)