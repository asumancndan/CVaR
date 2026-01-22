import logging
from typing import Tuple, Union
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_data(
    stocks: list[str],
    start:  dt.datetime,
    end: dt.datetime
) -> Tuple[pd.DataFrame, pd. Series, pd.DataFrame]:
    """
    Fetch historical stock data from Yahoo Finance and compute statistics. 

    Args:
        stocks:  List of stock ticker symbols (e.g., ['CBA. AX', 'BHP.AX'])
        start: Start date for historical data
        end: End date for historical data

    Returns:
        Tuple of (returns DataFrame, mean returns Series, covariance matrix)

    Raises:
        ValueError:  If stocks list is empty or dates are invalid
        Exception: If data fetch fails
    """
    if not stocks or len(stocks) == 0:
        raise ValueError("Stocks list cannot be empty")

    if start >= end:
        raise ValueError("Start date must be before end date")

    try:
        logger.info(f"Fetching data for {len(stocks)} stocks from {start. date()} to {end.date()}")
        stock_data = pdr.get_data_yahoo(stocks, start=start, end=end)
        stock_data = stock_data["Close"]

        if stock_data.empty:
            raise ValueError("No data retrieved for the specified period")

        returns = stock_data.pct_change()
        mean_returns = returns. mean()
        cov_matrix = returns.cov()

        logger.info(f"Successfully retrieved {len(stock_data)} trading days of data")
        return returns, mean_returns, cov_matrix

    except Exception as e: 
        logger.error(f"Failed to fetch data: {str(e)}")
        raise


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd. DataFrame,
    time_periods: int = 252
) -> Tuple[float, float]: 
    """
    Calculate portfolio returns and standard deviation.

    Args:
        weights: Portfolio weights (must sum to 1)
        mean_returns: Mean daily returns for each asset
        cov_matrix:  Covariance matrix of returns
        time_periods: Number of trading periods per year (default: 252)

    Returns:
        Tuple of (annualized returns, annualized standard deviation)

    Raises:
        ValueError:  If weights don't sum to approximately 1
    """
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1, got {np.sum(weights)}")

    if len(weights) != len(mean_returns):
        raise ValueError("Weights length must match returns length")

    returns = np.sum(mean_returns * weights) * time_periods
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(time_periods)

    return returns, std


def historical_var(
    returns: Union[pd.Series, pd.DataFrame],
    alpha: float = 5
) -> Union[float, pd.Series]:
    """
    Calculate Value at Risk (VaR) at a given confidence level using historical method.

    Args:
        returns: Series or DataFrame of returns
        alpha:  Confidence level (percentile), default 5 (95% confidence)

    Returns:
        VaR value (scalar for Series, Series for DataFrame)

    Raises:
        TypeError: If returns is not a Series or DataFrame
        ValueError: If alpha is not between 0 and 100
    """
    if not 0 <= alpha <= 100:
        raise ValueError("Alpha must be between 0 and 100")

    if isinstance(returns, pd.Series):
        var = np.percentile(returns, alpha)
        logger.debug(f"VaR at {alpha}% confidence:  {var:. 4f}")
        return var

    elif isinstance(returns, pd. DataFrame):
        return returns.aggregate(historical_var, alpha=alpha)

    else:
        raise TypeError("Expected returns to be DataFrame or Series")


def historical_cvar(
    returns:  Union[pd.Series, pd. DataFrame],
    alpha: float = 5
) -> Union[float, pd.Series]:
    """
    Calculate Conditional Value at Risk (CVaR) at a given confidence level.

    CVaR is the average of returns that fall below the VaR threshold. 

    Args:
        returns: Series or DataFrame of returns
        alpha:  Confidence level (percentile), default 5 (95% confidence)

    Returns:
        CVaR value (scalar for Series, Series for DataFrame)

    Raises:
        TypeError: If returns is not a Series or DataFrame
        ValueError:  If alpha is not between 0 and 100
    """
    if not 0 <= alpha <= 100:
        raise ValueError("Alpha must be between 0 and 100")

    if isinstance(returns, pd.Series):
        var_threshold = historical_var(returns, alpha=alpha)
        below_var = returns <= var_threshold
        cvar = returns[below_var].mean()
        logger.debug(f"CVaR at {alpha}% confidence:  {cvar:.4f}")
        return cvar

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historical_cvar, alpha=alpha)

    else:
        raise TypeError("Expected returns to be DataFrame or Series")


def main():
    """Main execution function demonstrating CVaR calculation."""
    try:
        # Configuration
        stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
        stocks = [stock + '.AX' for stock in stock_list]
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=800)

        # Fetch data
        returns, mean_returns, cov_matrix = get_data(stocks, start=start_date, end=end_date)
        returns = returns.dropna()

        if returns.empty:
            logger. error("No valid returns data after removing NaN values")
            return

        logger.info(f"Analysis period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Number of trading days: {len(returns)}")

        # Generate random portfolio
        weights = np.random.random(len(returns. columns))
        weights /= np.sum(weights)

        logger.info(f"Portfolio allocation: {dict(zip(stocks, weights))}")

        # Calculate portfolio returns
        returns['portfolio'] = returns.dot(weights)

        # Calculate performance metrics
        portfolio_return, portfolio_std = portfolio_performance(
            weights, mean_returns, cov_matrix
        )

        # Calculate risk metrics
        var_95 = historical_var(returns['portfolio'], alpha=5)
        cvar_95 = historical_cvar(returns['portfolio'], alpha=5)

        # Display results
        logger.info("="*60)
        logger.info("PORTFOLIO ANALYSIS RESULTS")
        logger.info("="*60)
        logger.info(f"Annualized Return: {portfolio_return:.4f}")
        logger.info(f"Annualized Std Dev: {portfolio_std:.4f}")
        logger.info(f"Sharpe Ratio (assuming 0% risk-free rate): {portfolio_return/portfolio_std:.4f}")
        logger.info(f"VaR (95% confidence): {var_95:.4f}")
        logger.info(f"CVaR (95% confidence): {cvar_95:.4f}")
        logger.info("="*60)

        return returns, mean_returns, cov_matrix

    except Exception as e: 
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
