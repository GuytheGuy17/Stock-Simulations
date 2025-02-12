#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:55:08 2025

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

#%%
# Monte Carlo simulation parameters
paths_mc = 10000
n_mc = 252  # Daily steps

# Monte Carlo Call Option Price
mc_call_price = BlackScholes.monte_carlo_simulation(
    S0=S0,
    K=K,
    T=T_option,
    r=r,
    sigma=sigma_option,
    paths=paths_mc,
    n=n_mc,
    option_type="call"
)

# Monte Carlo Put Option Price
mc_put_price = BlackScholes.monte_carlo_simulation(
    S0=S0,
    K=K,
    T=T_option,
    r=r,
    sigma=sigma_option,
    paths=paths_mc,
    n=n_mc,
    option_type="put"
)

print(f"Monte Carlo Call Option Price: ${mc_call_price:.2f}")
print(f"Monte Carlo Put Option Price: ${mc_put_price:.2f}")



#%%

# CIR Model Parameters
r0 = 0.05       # Initial interest rate (5%)
kappa = 0.15    # Mean reversion rate
theta = 0.05    # Long-term mean interest rate
sigma_r = 0.02  # Volatility of interest rate
T_cir = 1       # Time horizon (1 year)
n_cir = 252     # Daily steps
paths_cir = paths  # Same number of paths as stock simulation

# Initialize the CIR model
cir_model = CoxIngersollRossModel(r0, kappa, theta, sigma_r, T_cir, n_cir, paths_cir)

# Simulate interest rate paths
interest_rate_paths = cir_model.simulate()

# Validate the simulated rates
print(f"Mean of simulated interest rates: {interest_rate_paths.mean():.4f}")
print(f"Minimum simulated interest rate: {interest_rate_paths.min():.4f}")
print(f"Maximum simulated interest rate: {interest_rate_paths.max():.4f}")

# Plot a subset of the simulated interest rate paths
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(interest_rate_paths[i], lw=1)
plt.title('Simulated Interest Rate Paths using CIR Model')
plt.xlabel('Time Steps (Days)')
plt.ylabel('Interest Rate')
plt.grid(True)
plt.show()

# %%
# Generate CIR Simulated Rates for Option Pricing
cir_model = CoxIngersollRossModel(r0=r0, kappa=kappa, theta=theta, sigma=sigma_r, T=T_option, n=n_mc, paths=paths_mc)
interest_rate_paths = cir_model.simulate()

# Validate the simulated rates
print(f"Mean of simulated interest rates for option pricing: {interest_rate_paths.mean():.4f}")
print(f"Minimum simulated interest rate for option pricing: {interest_rate_paths.min():.4f}")
print(f"Maximum simulated interest rate for option pricing: {interest_rate_paths.max():.4f}")

# Monte Carlo Simulation for Option Pricing with CIR Rates
def monte_carlo_option_with_cir_corrected(S0, K, T, sigma, stock_paths, rate_paths, dt, option_type="call"):
    """
    Monte Carlo simulation for option pricing using CIR interest rates with correct discounting.
    :param S0: Initial stock price
    :param K: Strike price
    :param T: Time to maturity
    :param sigma: Volatility
    :param stock_paths: Simulated stock price paths
    :param rate_paths: Simulated interest rate paths (CIR model)
    :param dt: Time step size
    :param option_type: 'call' or 'put'
    :return: Simulated option price
    """
    # Calculate the cumulative discount factor for each path
    cumulative_rates = np.sum(rate_paths * dt, axis=1)  # Integral of r(t) dt
    discount_factors = np.exp(-cumulative_rates)
    
    # Final stock prices
    final_prices = stock_paths[:, -1]
    
    # Payoff based on option type
    if option_type == "call":
        payoff = np.maximum(0, final_prices - K)
    elif option_type == "put":
        payoff = np.maximum(0, K - final_prices)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Discount payoffs using the cumulative discount factor
    discounted_payoff = payoff * discount_factors
    
    # Return the average discounted payoff
    return np.mean(discounted_payoff)

# Simulate Stock Price Paths with Risk-Neutral Drift
np.random.seed(42)
stock_paths = np.zeros((paths_mc, n_mc))
stock_paths[:, 0] = S0

for t in range(1, n_mc):
    # Correct Drift Term
    drift = (interest_rate_paths[:, t-1] - 0.5 * sigma**2) * dt
    # Correct Diffusion Term
    diffusion = sigma * np.sqrt(dt) * Z[:, t-1]
    # Update Stock Prices
    stock_paths[:, t] = stock_paths[:, t-1] * np.exp(drift + diffusion)

# Plot a subset of the simulated stock price paths
plt.figure(figsize=(12, 6))
for i in range(100):
    plt.plot(stock_paths[i], lw=1)
plt.title(f'Simulated {ticker} Stock Price Paths over {T} Year(s) with CIR Rates')
plt.xlabel('Time Steps (Days)')
plt.ylabel('Stock Price ($)')
plt.grid(True)
plt.show()

# Calculate Option Prices with Correct Discounting
cir_mc_call_price = monte_carlo_option_with_cir_corrected(
    S0=S0,
    K=K,
    T=T_option,
    sigma=sigma_option,
    stock_paths=stock_paths,
    rate_paths=interest_rate_paths,
    dt=dt,
    option_type="call"
)

cir_mc_put_price = monte_carlo_option_with_cir_corrected(
    S0=S0,
    K=K,
    T=T_option,
    sigma=sigma_option,
    stock_paths=stock_paths,
    rate_paths=interest_rate_paths,
    dt=dt,
    option_type="put"
)

# Display Results
print(f"Monte Carlo Call Option Price with CIR Rates: ${cir_mc_call_price:.2f}")
print(f"Monte Carlo Put Option Price with CIR Rates: ${cir_mc_put_price:.2f}")
