# Stock-Simulations

# Financial Option Pricing and Risk Management

Welcome to the **Financial Option Pricing and Risk Management** project repository. This project provides a comprehensive toolkit for pricing European options, simulating stock and interest rate paths, and implementing risk management strategies such as delta hedging. Leveraging established financial models and numerical methods, the code offers both analytical and simulation-based approaches to option valuation.

## Table of Contents

- [Introduction](#introduction)
- [Models and Theories](#models-and-theories)
  - [Black-Scholes Model](#black-scholes-model)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Cox-Ingersoll-Ross (CIR) Model](#coxingersoll-ross-cir-model)
  - [Delta Hedging](#delta-hedging)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Retrieval and Visualization](#data-retrieval-and-visualization)
  - [Stock Price Simulation](#stock-price-simulation)
  - [Option Pricing](#option-pricing)
  - [Interest Rate Simulation](#interest-rate-simulation)
  - [Option Pricing with Stochastic Interest Rates](#option-pricing-with-stochastic-interest-rates)
  - [Delta Hedging Simulation](#delta-hedging-simulation)
- [Results](#results)
- [References](#references)
- [License](#license)

## Introduction

Options are fundamental financial derivatives that provide the right, but not the obligation, to buy or sell an asset at a predetermined price within a specified timeframe. Accurately pricing these options is crucial for traders, risk managers, and financial analysts. This project implements key financial models and simulation techniques to price European call and put options, simulate stock and interest rate paths, and evaluate hedging strategies.

## Models and Theories

### Black-Scholes Model

The **Black-Scholes model** is a mathematical framework for pricing European-style options. It assumes that the underlying asset follows a geometric Brownian motion with constant drift and volatility. The model provides closed-form solutions for option prices, facilitating quick and efficient valuation.

**Key Components:**

- **Spot Price (S):** Current price of the underlying asset.
- **Strike Price (K):** Predefined price at which the option can be exercised.
- **Time to Maturity (T):** Time remaining until the option expires.
- **Risk-Free Rate (r):** Theoretical rate of return of an investment with zero risk.
- **Volatility (σ):** Measure of the asset's price fluctuations.

**Formulas:**

For a **Call Option**:
\[ C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2) \]

For a **Put Option**:
\[ P = K e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1) \]

Where:
\[ d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} \]
\[ d_2 = d_1 - \sigma \sqrt{T} \]
\[ N(\cdot) \] is the cumulative distribution function of the standard normal distribution.

### Monte Carlo Simulation

**Monte Carlo simulation** is a numerical method used to estimate the value of options by simulating a large number of possible price paths for the underlying asset. It is particularly useful for pricing complex derivatives where analytical solutions may not exist.

**Process:**

1. **Simulate Stock Paths:** Generate numerous possible future paths of the stock price based on geometric Brownian motion.
2. **Calculate Payoffs:** For each simulated path, compute the option's payoff at maturity.
3. **Discount Payoffs:** Discount the average of these payoffs back to present value using the risk-free rate.

### Cox-Ingersoll-Ross (CIR) Model

The **Cox-Ingersoll-Ross (CIR) model** is used to model the evolution of interest rates over time. Unlike models that assume constant interest rates, the CIR model incorporates mean reversion, where interest rates tend to move towards a long-term average.

**Key Components:**

- **Initial Rate (r₀):** Starting interest rate.
- **Mean Reversion Rate (κ):** Speed at which rates revert to the mean.
- **Long-Term Mean (θ):** The average rate to which interest rates revert.
- **Volatility (σ):** Variability in interest rate movements.

**Stochastic Differential Equation:**
\[ dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t \]

Where:
- \( dr_t \) is the change in interest rate.
- \( dW_t \) is a Wiener process (standard Brownian motion).

### Delta Hedging

**Delta hedging** is a risk management strategy that aims to reduce the sensitivity of an option's price to small changes in the price of the underlying asset. By dynamically adjusting the position in the underlying asset, traders can neutralize the option's delta (rate of change of the option price with respect to the underlying asset price).

**Process:**

1. **Calculate Delta:** Determine the option's delta using the Black-Scholes formula.
2. **Adjust Position:** Buy or sell the underlying asset to match the delta, ensuring the portfolio is hedged.
3. **Rebalance:** Continuously adjust the hedging position as the underlying asset price and time to maturity change.

## Code Structure

The project is organized into the following components:

- **Classes:**
  - `BlackScholes`: Implements the Black-Scholes option pricing formula and Monte Carlo simulation methods.
  - `CoxIngersollRossModel`: Simulates interest rate paths using the CIR model.

- **Data Retrieval:**
  - Fetches historical stock data using the `yfinance` library.

- **Simulations:**
  - Stock price simulations using geometric Brownian motion.
  - Interest rate simulations using the CIR model.
  - Option pricing using both analytical and simulation-based methods.
  - Delta hedging strategy implementation and evaluation.

- **Visualization:**
  - Plots historical and simulated stock prices.
  - Visualizes interest rate paths and hedging performance.

## Installation

To get started with this project, ensure you have Python 3.7 or later installed. Follow the steps below to set up the environment:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/financial-option-pricing.git
   cd financial-option-pricing

