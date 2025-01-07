# Stock-Simulations

This project implements fundamental financial models and strategies to price European options, simulate interest rate dynamics, and manage risk through delta hedging. The primary models included are the Black-Scholes model, an extension of the Black-Scholes model incorporating the Cox-Ingersoll-Ross (CIR) interest rate model, and a delta hedging strategy.

## Table of Contents

- [Introduction](#introduction)
- [Models and Strategies](#models-and-strategies)
  - [Black-Scholes Model](#black-scholes-model)
  - [Black-Scholes Model with CIR Interest Rates](#black-scholes-model-with-cir-interest-rates)
  - [Delta Hedging](#delta-hedging)

## Introduction

Options are versatile financial instruments that grant the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified timeframe. Accurate option pricing is essential for traders, investors, and risk managers to make informed decisions. This project leverages well-established financial models and numerical methods to price European call and put options, simulate interest rate movements, and implement risk management strategies.

## Models and Strategies

### Black-Scholes Model

The **Black-Scholes model** is a pioneering mathematical framework for option pricing. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, it provides a closed-form solution to determine the fair price of European-style options.

**Key Assumptions:**
- The underlying asset price follows a geometric Brownian motion with constant drift and volatility.
- No dividends are paid out during the option's life.
- Markets are efficient, and there are no transaction costs.
- The risk-free interest rate is constant and known.
- The option can only be exercised at maturity (European option).

**Basic Formula:**

For a **Call Option**:
$$'
C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)
$$'

For a **Put Option**:
$$'
P = K e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
$$'

Where:
$$'
d_1 = \frac{\ln{\left(\frac{S}{K}\right)} + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}
$$'
$$'
d_2 = d_1 - \sigma \sqrt{T}
$$'
- \(' S '\) = Current stock price
- \(' K '\) = Strike price
- \( T \) = Time to maturity (in years)
- \( r \) = Risk-free interest rate
- \( \sigma \) = Volatility of the underlying asset
- \( N(\cdot) \) = Cumulative distribution function of the standard normal distribution

### Black-Scholes Model with CIR Interest Rates

While the standard Black-Scholes model assumes a constant risk-free interest rate, real-world scenarios often involve fluctuating interest rates. To account for this, the **Black-Scholes model with Cox-Ingersoll-Ross (CIR) interest rates** integrates the CIR model to simulate dynamic interest rate paths.

**Cox-Ingersoll-Ross (CIR) Model:**

The CIR model is a popular choice for modeling the evolution of interest rates due to its mean-reverting properties and ensuring that interest rates remain positive.

**Key Components:**
- **Initial Rate (\( r_0 \))**: The starting interest rate.
- **Mean Reversion Rate (\( \kappa \))**: The speed at which interest rates revert to the long-term mean.
- **Long-Term Mean (\( \theta \))**: The equilibrium interest rate level.
- **Volatility (\( \sigma \))**: The volatility of interest rate movements.

**Stochastic Differential Equation:**
$$
dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t
$$
- \( dr_t \) = Change in interest rate
- \( dW_t \) = Wiener process (standard Brownian motion)

By simulating multiple interest rate paths using the CIR model, the extended Black-Scholes framework can incorporate the variability in discount rates, leading to more accurate option pricing under fluctuating economic conditions.

### Delta Hedging

**Delta hedging** is a risk management strategy used to mitigate the directional risk associated with price movements of the underlying asset. It involves adjusting the position in the underlying asset to maintain a delta-neutral portfolio.

**Key Concepts:**
- **Delta (\( \Delta \))**: Measures the sensitivity of the option's price to changes in the price of the underlying asset.
- **Delta-Neutral Portfolio**: A portfolio where the overall delta is zero, meaning it is insensitive to small price movements in the underlying asset.

**Strategy:**
1. **Calculate Delta**: Using the Black-Scholes model, determine the delta of the option.
2. **Adjust Position**: Buy or sell a quantity of the underlying asset equal to the delta to offset the option's price sensitivity.
3. **Rebalance**: Continuously update the hedge as the delta changes with the underlying asset's price and as time progresses.

**Benefits:**
- Reduces the risk of adverse price movements.
- Allows for better management of the portfolio's sensitivity to market changes.

**Challenges:**
- Requires continuous monitoring and adjustment.
- Transaction costs can accumulate due to frequent rebalancin
