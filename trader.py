import random
from typing import List

import mesa
import numpy as np
from helper import getRandomUniform, lognormal, normal
from limit_order import LimitOrder, OrderStatus


class Trader(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pnl = 0  # Profit and Loss
        self.type = "default"  # Example type
        self.init_wealth = round(self.model.wealth_min * (1 + np.random.pareto(self.model.wealth_alpha)))
       
        # risk aversion
        self.risk_aversion = round(normal(0.5, 0.1), 2)
        self.risk_appetit = round(1 - self.risk_aversion, 2)

        # starting portfolio allocation
        self.starting_share_alloc_cash = (self.risk_appetit + getRandomUniform(-0.05, 0.05)) * self.init_wealth
        self.num_of_shares = round(self.starting_share_alloc_cash / self.model.initial_market_price)
        self.cash = self.init_wealth - self.num_of_shares * self.model.initial_market_price
        self.orders: List[LimitOrder] = [] # List of LimitOrder objects

        # skill
        self.skill = round(normal(0.5, 0.2), 2)

        # valuation bias
        self.valuation_bias = round(1 + normal(0.05 - 0.05 * self.skill, 0.10), 4)

        # overconfidence
        self.overconfidence = round(normal(0.5 - 0.5 * self.skill / 2, 0.15), 2)

        # greek alpha
        self.estimated_true_value = self.model.initial_market_price * (1 + self.valuation_bias)

        # self.placed_orders = List[LimitOrder] = []  # List of orders placed by the trader
        
        # print(f"""Trader #{self.unique_id!s}
        #       ================================= 
        #       Cash: {self.cash}, 
        #       Num of shares: {self.num_of_shares},
        #       Risk aversion: {self.risk_aversion}, risk appetite: {self.risk_appetit}, 
        #       Valuation Bias: {self.valuation_bias}, 
        #       Overconfidence: {self.overconfidence},
        #       Skill: {self.skill}, 
        #       =================================
        #      """)
        
    def get_total_wealth(self):
        """
        Returns the total wealth of the trader.
        """
        
        return self.cash + self.num_of_shares * self.model.get_market_price() 
    
    def get_decision_threshold(self):
        # overconfidence lowers decision treshold
        return self.model.decision_threshold - max(0, self.overconfidence - 0.5)

    def step(self):
        # print(f"Trader {self.unique_id} starting step...")
        self.estimate_true_value()
        self.decide_to_trade()
        # print(f"Trader {self.unique_id} completed step.")

    def estimate_true_value(self):
        # Example logic for estimating the true value
        noise_abs = self.model.get_market_price() * self.model.noise_range
        uninformed_estimation = round(max(0.01, self.estimated_true_value + getRandomUniform(-noise_abs, noise_abs)), 2)
        
        tilt = self.model.get_lob_tilt() 
        if tilt != 0:
            tilt_abs = tilt / 10 * self.model.get_market_price()     
            self.estimated_true_value = round(uninformed_estimation + tilt_abs * getRandomUniform(0.5, 1), 2)

        else:
            self.estimated_true_value = round(uninformed_estimation, 2)

        # print(f"Trader {self.unique_id} estimated true value: {self.estimated_true_value}")

    def get_portfolio_share_stocks(self):
        """
        Returns the portfolio share of stocks.
        """
        return (self.num_of_shares * self.model.get_market_price()) / self.get_total_wealth()
    
    def has_active_orders(self):
        """
        Returns True if the trader has active orders, False otherwise.
        """
        return any(order.get_status() == OrderStatus.OPEN or order.get_status() == OrderStatus.PARTIALLY_FILLED for order in self.orders)

    def decide_to_trade(self):

        if not getRandomUniform(0, 1) <= self.overconfidence:
            return
        
        if self.has_active_orders():
            return

        decision = 0

        portfolio_share_stocks = self.get_portfolio_share_stocks()
        diff_portfolio_alloc = self.risk_appetit - portfolio_share_stocks
        rel_diff_portfolio_alloc = diff_portfolio_alloc / self.risk_appetit
        # print(f"Trader {self.unique_id} portfolio share stocks: {portfolio_share_stocks:.2%}, diff wrt risk appetite: {diff_relative:.2%}")

        if abs(rel_diff_portfolio_alloc) > self.get_decision_threshold():
            # print(f"Trader {self.unique_id} is considering selling shares based on portfolio allocation...")
            decision += rel_diff_portfolio_alloc
            

        rel_deviation_price_estimation = (self.estimated_true_value - self.model.get_market_price()) / self.model.get_market_price()
        # print(f"Trader {self.unique_id} relative deviation price estimation: {rel_deviation_price_estimation:.2%}")
        if abs(rel_deviation_price_estimation) > self.get_decision_threshold():
            decision += rel_deviation_price_estimation
            
        # Example decision logic
        if abs(decision) > self.get_decision_threshold():
            direction = "buy" if decision > 0 else "sell"
            order_size = self.get_order_size(direction)
            if abs(order_size) > 0:
                # print(f"Trader {self.unique_id} is placing a {direction} order of size {order_size}...")
                order = LimitOrder(
                    trader_id=self.unique_id,
                    price=self.get_limit_price(direction),
                    quantity=order_size,
                    trading_day=self.model.current_day
                )
                self.orders.append(order)
                self.model.process_order(order)
        # else:
            # print(f"Trader {self.unique_id} decided not to trade.")


    def get_order_size(self, direction):
        diff_portfolio_alloc = self.risk_appetit - self.get_portfolio_share_stocks()
        num_of_stocks = round(diff_portfolio_alloc * self.get_total_wealth() / self.model.get_market_price())
        

        # Ensure the order size is within the available cash
        if direction == "buy":
            num_of_stocks = round(self.cash / self.model.get_market_price())
        else:
            num_of_stocks = round(self.num_of_shares)
       
        # Adjust the order size based on risk appetite
        if self.risk_appetit <= 0.1:
            num_of_stocks = max(1, round(num_of_stocks / 10))
        elif self.risk_appetit > 0.1 and self.risk_appetit <= 0.2:
            num_of_stocks = max(1, round(num_of_stocks / 8))
        elif self.risk_appetit > 0.2 and self.risk_appetit <= 0.3:
            num_of_stocks = max(1, round(num_of_stocks / 6))
        elif self.risk_appetit > 0.3 and self.risk_appetit <= 0.4:
            num_of_stocks = max(1, round(num_of_stocks / 5))
        elif self.risk_appetit > 0.4 and self.risk_appetit <= 0.5:
            num_of_stocks = max(1, round(num_of_stocks / 4))
        elif self.risk_appetit > 0.5 and self.risk_appetit <= 0.6:
            num_of_stocks = max(1, round(num_of_stocks / 3.5))
        elif self.risk_appetit > 0.6 and self.risk_appetit <= 0.7:
            num_of_stocks = max(1, round(num_of_stocks / 3))
        elif self.risk_appetit > 0.7 and self.risk_appetit <= 0.8:
            num_of_stocks = max(1, round(num_of_stocks / 2.5))
        elif self.risk_appetit > 0.8 and self.risk_appetit <= 0.9:
            num_of_stocks = max(1, round(num_of_stocks / 2))
        elif self.risk_appetit > 0.9 and self.risk_appetit <= 1:
            num_of_stocks = max(1, round(num_of_stocks / 1.5))

        return abs(num_of_stocks) * (1 if direction == "buy" else -1)  # Return positive for buy, negative for sell


    def get_limit_price(self, order_type):
        """
        Returns the limit price for the order.
        """
        # Example logic for determining the limit price
        if order_type == "buy":
            safety_margin = self.estimated_true_value * max(0, normal(0.10, 0.1))
            return min(round(self.estimated_true_value - self.risk_aversion * safety_margin, 2), self.model.get_ask_quote_lob())
        else:
            safety_margin = self.estimated_true_value * max(0, normal(0.10, 0.1))
            return max(round(self.estimated_true_value + self.risk_aversion * safety_margin, 2), self.model.get_bid_quote_lob())


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



