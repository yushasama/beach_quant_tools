from typing import List, Set,Tuple
import numpy as np
import itertools

class ComputeUtils:
    """
    A utility class for performing generic computations.
    """
    
    def __init__(self) -> None:
        """
        Initializes the ComputeUtils class.
        """
        pass

    @staticmethod
    def normalize_weights(weights: List[float]) -> List[float]:
        """
        Normalizes a list of weights so that they sum to 1.

        Args:
            weights (List[float]): A list of weights.

        Returns:
            List[float]: A list of normalized weights.
        """
        weights = np.array(weights)
        
        sum_weights = np.sum(weights)
        
        normalized_weights = weights / sum_weights

        return normalized_weights.tolist()

    @staticmethod
    def standard_deviation(input: List[float], rounded: bool = True, decimals: int = 4) -> float:
        """
        Calculates the standard deviation of a list of numbers.

        Args:
            input (List[float]): A list of numbers.
            rounded (bool): Whether to round the result. Defaults to True.
            decimals (int): Number of decimal places to round to if rounded is True. Defaults to 2.

        Returns:
            float: The standard deviation of the input list.
        """
        std_dev = np.std(input) 
        
        if rounded:
            return round(std_dev, decimals)
        else:
            return std_dev
    
    @staticmethod
    def cartesian_product(list_a: List[any], list_b: List[any]) -> Set[Tuple[any, any]]:
        """
        Generate the Cartesian product of two lists.

        Parameters:
        list_a (List[any]): The first list.
        list_b (List[any]): The second list.

        Returns:
        Set[Tuple[any, any]]: A set containing tuples, each representing a pair from the Cartesian product of the input lists.
        """
        
        return set(itertools.product(list_a, list_b))
    
    @staticmethod
    def covariance_matrix(list_a: List[float], list_b: List[float]) -> np.ndarray:
        """
        Calculate the covariance matrix from two lists of floats.

        Parameters:
        list_a (List[float]): The first list of floats.
        list_b (List[float]): The second list of floats.

        Returns:
        np.ndarray: A nxn covariance matrix.
        """

        if len(list_a) != len(list_b):
            raise ValueError(f"The input sizes of 'list_a' and 'list_b' must be equal. 'list_a' has input size {len(list_a)} while 'list_b' has input size {len(list_b)}")

        cartesian_product = ComputeUtils.cartesian_product(list_a, list_b)

        i_entries = [pair[0] for pair in cartesian_product]
        j_entries = [pair[1] for pair in cartesian_product]

        covariance_matrix = np.array([i_entries, j_entries])

        return np.cov(covariance_matrix)

    @staticmethod
    def portfolio_variance(asset_returns: List[float], weights: List[float]) -> float:
        """
        Calculate the portfolio variance given asset returns and weights.

        Parameters:
        asset_returns (List[float]): A list of asset returns.
        weights (List[float]): A list of weights for each asset.

        Returns:
        float: The variance of the portfolio.

        Formula (Latex):
         The portfolio variance dobule formula can be represented as a matrix transformation:
         
         \begin{}

        \[
        \sigma^2_p = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} = w
        \]

        """

        covariance_matrix = ComputeUtils.covariance_matrix(asset_returns, asset_returns)

        w = np.array(weights).reshape(1,-1)

        # Creates a 1x1 matrix of a scalar float value, access its value with [0,0]
        portfolio_variance = np.dot(w.T, np.dot(covariance_matrix, w))[0,0]

        return portfolio_variance