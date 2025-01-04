#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:54:18 2025

@author: guymcclennan
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

class BlackScholes:
    @staticmethod
    def option_price(S, K, T, r, sigma, option_type="call"):
        """
        Calculate the Black-Scholes price for European options.
        :param S: Spot price
        :param K: Strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate
        :param sigma: Volatility
        :param option_type: 'call' or 'put'
        :return: Option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    @staticmethod
    def monte_carlo_simulation(S0, K, T, r, sigma, paths, n, option_type="call"):
        """
        Monte Carlo simulation for option pricing using Black-Scholes.
        :param S0: Initial stock price
        :param K: Strike price
        :param T: Time to maturity
        :param r: Risk-free interest rate
        :param sigma: Volatility
        :param paths: Number of simulation paths
        :param n: Number of time steps
        :param option_type: 'call' or 'put'
        :return: Simulated option prices
        """
        dt = T / n
        S = np.zeros((paths, n))
        S[:, 0] = S0
        for t in range(1, n):
            z = np.random.standard_normal(paths)
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        payoff = np.maximum(0, S[:, -1] - K if option_type == "call" else K - S[:, -1])
        return np.exp(-r * T) * np.mean(payoff)
    

class CoxIngersollRossModel:
    def __init__(self, r0, kappa, theta, sigma, T, n, paths):
        """
        Initialize the Cox-Ingersoll-Ross model parameters.
        :param r0: Initial interest rate
        :param kappa: Rate of mean reversion
        :param theta: Long-term mean interest rate
        :param sigma: Volatility of interest rate
        :param T: Time horizon (in years)
        :param n: Number of time steps
        :param paths: Number of simulation paths
        """
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.n = n
        self.paths = paths

    def simulate(self):
        """
        Simulate interest rate paths using the CIR model.
        :return: Simulated interest rate paths
        """
        dt = self.T / self.n
        r = np.zeros((self.paths, self.n))
        r[:, 0] = self.r0

        for t in range(1, self.n):
            dr = (
                self.kappa * (self.theta - r[:, t-1]) * dt
                + self.sigma * np.sqrt(np.maximum(r[:, t-1], 0) * dt)
                * np.random.normal(size=self.paths)
            )
            r[:, t] = np.maximum(r[:, t-1] + dr, 0)

        return r
    
#%%

# Define the ticker symbol and the time period
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = '2023-12-31'

# Fetch the data
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows
print(data.head())

#%%
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price')
plt.title(f'Historical Closing Prices of {ticker}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Drop the first row with NaN
data = data.dropna()

# Display statistics
print(data['Daily Return'].describe())
#%%
# Calculate average daily return and daily volatility
mu_daily = data['Daily Return'].mean()
sigma_daily = data['Daily Return'].std()

print(f"Estimated Daily Drift (mu): {mu_daily}")
print(f"Estimated Daily Volatility (sigma): {sigma_daily}")
# Number of trading days in a year
trading_days = 252

# Annualized drift and volatility
mu = mu_daily * trading_days
sigma = sigma_daily * np.sqrt(trading_days)

print(f"Annualized Drift (mu): {mu}")
print(f"Annualized Volatility (sigma): {sigma}")

#%%
# Simulation parameters
S0 = data['Close'].iloc[-1]  # Last closing price as initial price
T = 1                    # Time horizon in years
n = 252                  # Number of time steps (daily)
dt = T / n               # Time step
paths = 1000             # Number of simulation paths

#%%

# Simulate stock price paths
np.random.seed(42)  # For reproducibility
Z = np.random.standard_normal((paths, n))
S = np.zeros((paths, n))
S[:, 0] = S0

for t in range(1, n):
    S[:, t] = S[:, t-1] * np.exp((mu / trading_days - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z[:, t-1])

# Plot a subset of the simulated paths
plt.figure(figsize=(12, 6))
for i in range(100):
    plt.plot(S[i], lw=1)
plt.title(f'Simulated {ticker} Stock Price Paths over {T} Year(s)')
plt.xlabel('Time Steps (Days)')
plt.ylabel('Stock Price ($)')
plt.grid(True)
plt.show()

#%%
# Option parameters
K = 200         # Strike price
T_option = 1      # Time to maturity in years
r = 0.05          # Risk-free interest rate (5%)
sigma_option = sigma  # Annualized volatility from simulation

# Assuming the BlackScholes class is already defined and imported

call_price = BlackScholes.option_price(S=S0, K=K, T=T_option, r=r, sigma=sigma_option, option_type="call")
if isinstance(call_price, pd.Series):
    call_price = call_price.iloc[0]  # Extract scalar value if it's a Series

put_price = BlackScholes.option_price(S=S0, K=K, T=T_option, r=r, sigma=sigma_option, option_type="put")
if isinstance(put_price, pd.Series):
    put_price = put_price.iloc[0]

print(f"Black-Scholes Call Option Price: ${call_price:.2f}")
print(f"Black-Scholes Put Option Price: ${put_price:.2f}")