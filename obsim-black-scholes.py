import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

# Input parameters 
r = 0        # risk-free interest rate
S0 = 100     # Stock price 
T = 0.5      # Time to maturity (years)
sigma = 0.2  # Volatility
n = 10000   # Simulations


class BlackScholes:    
    def __init__(self, K): 
        self.K = K
        
        self.d1 = (np.log(S0/self.K) + (r + ((sigma**2)/2))*T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        
    def call(self):
        bs_call_price = (S0 * norm.cdf(self.d1, 0, 1)) - (self.K * np.exp(-r*T) * norm.cdf(self.d2, 0, 1))
        
        return round(bs_call_price, 4)
        
    def put(self):
        bs_put_price = self.K * np.exp(r*T) * norm.cdf(-self.d2, 0 , 1) - S0 * norm.cdf(-self.d1, 0, 1)
        
        return round(bs_put_price, 4)
      
      
class MonteCarlo:
    def __init__(self, n):
        self.n = n
        self.bs_dictionary = self.generate_dictionary()
        
    def generate_dictionary(self):
        # Creating a list of 150 strike prices, evenly spaced, in the interval (0.5 * S0, 1.5 * S0)
        strike_list = np.linspace(0.5 * S0, 1.5 * S0, 150)
        
        # Creating a list of black scholes call and put prices for each respective strike price
        bs_call_prices = [BlackScholes(K).call() for K in strike_list]
        bs_put_prices = [BlackScholes(K).put() for K in strike_list]

        # Combining the black scholes call and put prices with the strike prices into a dictionary
        return {(c,p): k for c, p, k in zip(bs_call_prices, bs_put_prices, strike_list)}
    
    def loss_minimizer(self, a):
        
        min_pair = None
        min_val = float('inf')
        
        for pair in self.bs_dictionary.keys():
            bs_call_price, bs_put_price = pair
            loss = (a * np.exp(r*T) * bs_call_price) + ((1 - a) * np.exp(r*T) * bs_put_price)
            
            if loss < min_val:
                min_val = loss
                min_pair = pair
        
        return round(self.bs_dictionary[min_pair], 2)
    
    def run_simulations(self):
        return [self.loss_minimizer(random.uniform(0, 1)) for m in range(self.n)]

      
simulated_stock_data = MonteCarlo(n).run_simulations()


class ObSim:
    def __init__(self, K):
        self.K = K
        
    def call(self):
        payoff_list = [max(ST - self.K, 0) for ST in simulated_stock_data]
        obsim_price = np.exp(-r*T) * (sum(payoff_list) / len(simulated_stock_data))
        
        return round(obsim_price, 4)
        
    def put(self):
        payoff_list = [max(self.K - ST, 0) for ST in simulated_stock_data]
        obsim_price = np.exp(-r*T) * (sum(payoff_list) / len(simulated_stock_data))
        
        return round(obsim_price, 4)
      
      
sample_strikes = [94, 96, 98, 100, 102, 104, 106]

# Creating Black Scholes call prices for the sample strikes [94, 106]
bs_sample_values = [BlackScholes(K).call() for K in sample_strikes]

# Creating Obsim call prices for the sample strikes [94, 106]
obsim_sample_values = [ObSim(K).call() for K in sample_strikes]

df_1 = pd.DataFrame(sample_strikes, columns=['Strike, K'])
df_1['BS Formula Price'] = bs_sample_values
df_1['ObSim Price'] = obsim_sample_values

