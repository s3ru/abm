import random

import mesa
import numpy as np
from helper import getRandomUniform, lognormal, normal


class Trader(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.cash = 100  # Example initial cash
        self.pnl = 0  # Profit and Loss
        self.type = "default"  # Example type
        self.cash = round(self.model.wealth_min * (1 + np.random.pareto(self.model.wealth_alpha)))

        # risk aversion
        self.risk_aversion = round(normal(0.5, 0.1), 2)
        self.risk_appetit = round(1 - self.risk_aversion, 2)

        self.starting_share_alloc_cash = (self.risk_appetit + getRandomUniform(-0.1, 0.1)) * self.cash

        self.num_of_shares = round(self.starting_share_alloc_cash / self.model.get_market_price())

        self.wealth = self.cash + self.num_of_shares * self.model.get_market_price()  # Example wealth calculation

        # skill
        self.skill = round(normal(0.5, 0.2), 2)

        # valuation bias
        self.vb = round(1 + normal(0.1 - 0.1 * self.skill, 0.05), 4)

        # overconfidence
        self.oc = round(normal(0.5 - 0.5 * self.skill, 0.05), 2)

        # greek alpha
        self.est_v = 0
        
        print(f"""Trader #{self.unique_id!s}
              ================================= 
              Cash: {self.cash}, 
              Num of shares: {self.num_of_shares},
              Risk aversion: {self.risk_aversion}, risk appetite: {self.risk_appetit}, 
              Valuation Bias: {self.vb}, 
              Overconfidence: {self.oc},
              Skill: {self.skill}, 
              =================================
             """)

    def step(self):
        print(f"Trader {self.unique_id} starting step...")
        self.estimate_true_value()
        print(f"Trader {self.unique_id} completed step.")

    def estimate_true_value(self):
        # Example logic for estimating the true value
        self.est_v = self.model.current_v * self.vb
        print(f"Trader {self.unique_id} estimated true value: {self.est_v}")

    # def make_decision(self):
    #     print(f"Trader {self.unique_id} is making a decision...")
    #     # Example decision logic
    #     if self.cash > 50:
    #         print(f"Trader {self.unique_id} has sufficient cash: {self.cash}. Considering placing a buy order.")
    #         self.place_order("buy")
    #     else:
    #         print(f"Trader {self.unique_id} has insufficient cash: {self.cash}. Considering placing a sell order.")
    #         self.place_order("sell")

    # def place_order(self, order_type):
    #     print(f"Trader {self.unique_id} placing a {order_type} order...")
    #     # Example order placement logic
    #     price = self.model.current_v * (1 + (0.01 if order_type == "buy" else -0.01))
    #     print(f"Trader {self.unique_id} {order_type} order price: {price}")
    #     self.model.lob.append((order_type, price, self))
    #     print(f"Trader {self.unique_id} {order_type} order added to the order book.")



