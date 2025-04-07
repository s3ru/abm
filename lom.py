# ABM einer Limit-Order-Markt-Simulation mit MESA
# Basierend auf der Beschreibung und orientiert an Goettler et al. (2009)
# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns

import mesa
from typing import List, Callable
from helper import lognormal, normal, poisson

from mesa.datacollection import DataCollector
import numpy as np
import random
import math

from limit_order import LimitOrder
from trader import Trader
from transaction import Transaction

class LimitOrderMarket(mesa.Model):
    def __init__(self, num_agents: int = 100, n_days: int = 100) -> None:
        super().__init__(seed=None)
        # wealth
        self.wealth_alpha = 2.0       # shape
        self.wealth_min = 1000.0       # minimum wealth

        self.num_agents: int = num_agents
        self.current_day: int = 0
            

        self.n_days: int = n_days
        self.current_v: float = 100.0
        self.initial_market_price: float = round(normal(self.current_v, 40))
        self.lob: List[LimitOrder] = []  # List of LimitOrder objects
        self.transactions: List[Transaction] = []  # List of transactions (price, volume)
        self.volumes: List[float] = []  # List of volumes
        self.data_collector = DataCollector(
            model_reporters={"MarketPrice": lambda m: m.get_market_price(),
                             "TrueValue": lambda m: m.current_v},
            agent_reporters={"PnL": "pnl"}
        )

        print(f"""LOM 
              ================================
                initial market price: {self.initial_market_price},
                current_v: {self.current_v}, 
                number of agents: {self.num_agents}, 
                number of days: {self.n_days}""")
      

        Trader.create_agents(model=self, n=num_agents)


    def step(self) -> None:
        print(f"Starting step {self.current_day}...")

        self.markov_price_step()

        # Fundamentalwert aktualisieren
        if poisson(0.1):
            shock = lognormal(0, 0.1)
            print(f"Shock applied: {shock}")
            self.current_v *= (1 + shock if random.random() < 0.5 else 1 - shock)
            print(f"Updated current_v: {self.current_v}")

        # Liquiditätsereignisse
        for agent in self.agents:
            if poisson(0.05):
                cash_change = normal(0, 10)
                agent.cash += cash_change
                print(f"Agent {agent.unique_id} cash updated by {cash_change}. New cash: {agent.cash}")

        self.agents.shuffle_do("step")

        self.data_collector.collect(self)

        self.current_day += 1
        print(f"Step {self.current_day} completed.")

    def get_market_price(self) -> float:
        if self.lob and len(self.lob) > 0:
            # Sort orders by price and then by timestamp
            self.lob.sort(key=lambda x: (x.price, x.timestamp))
            best_order = self.lob[0]
            print(f"Best order in LOB: {best_order}")
            return best_order.price
        else:
            return self.initial_market_price

    def evaluate_efficiency(self):
        prices = self.data_collector.get_model_vars_dataframe()["MarketPrice"].values
        true_vals = self.data_collector.get_model_vars_dataframe()["TrueValue"].values
        avg_deviation = np.mean(np.abs(prices - true_vals))
        print(f"Durchschnittliche Abweichung Marktpreis vs. fundamentaler Wert: {avg_deviation:.2f}")

    def markov_price_step(self):
        mu = 100
        phi = 0.95
        sigma = 1.0
        self.current_v = mu + phi * (self.current_v - mu) + np.random.normal(0, sigma)

    def run_model(self):
        for _ in range(self.n_days):
            self.step()

        self.evaluate_efficiency()
