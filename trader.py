import random
from typing import List

import mesa
import numpy as np
from helper import getRandomUniform, lognormal, normal
from limit_order import LimitOrder


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

        # skill
        self.skill = round(normal(0.5, 0.2), 2)

        # valuation bias
        self.valuation_bias = round(1 + normal(0.1 - 0.1 * self.skill, 0.05), 4)

        # overconfidence
        self.overconfidence = round(normal(0.5 - 0.5 * self.skill, 0.05), 2)

        # greek alpha
        self.estimated_true_value = 0

        # self.placed_orders = List[LimitOrder] = []  # List of orders placed by the trader
        
        print(f"""Trader #{self.unique_id!s}
              ================================= 
              Cash: {self.cash}, 
              Num of shares: {self.num_of_shares},
              Risk aversion: {self.risk_aversion}, risk appetite: {self.risk_appetit}, 
              Valuation Bias: {self.valuation_bias}, 
              Overconfidence: {self.overconfidence},
              Skill: {self.skill}, 
              =================================
             """)
        
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
        self.estimated_true_value = min(0.01, self.model.get_market_price() * self.valuation_bias)
        # print(f"Trader {self.unique_id} estimated true value: {self.estimated_true_value}")

    def get_portfolio_share_stocks(self):
        """
        Returns the portfolio share of stocks.
        """
        return (self.num_of_shares * self.model.get_market_price()) / self.get_total_wealth()

    def decide_to_trade(self):

        decision = self.valuation_bias - 1

        portfolio_share_stocks = self.get_portfolio_share_stocks()
        diff_portfolio_alloc = self.risk_appetit - portfolio_share_stocks
        diff_relative = diff_portfolio_alloc / self.risk_appetit
        # print(f"Trader {self.unique_id} portfolio share stocks: {portfolio_share_stocks:.2%}, diff wrt risk appetite: {diff_relative:.2%}")

        if abs(diff_relative) > self.get_decision_threshold():
            if diff_portfolio_alloc > 0:
                # print(f"Trader {self.unique_id} is considering selling shares based on portfolio allocation...")
                decision += diff_relative * -1
            else:
                # print(f"Trader {self.unique_id} is considering buying shares based on portfolio allocation...")
                decision += diff_relative

        rel_deviation_price_estimation = (self.estimated_true_value - self.model.get_market_price()) / self.model.get_market_price()
        # print(f"Trader {self.unique_id} relative deviation price estimation: {rel_deviation_price_estimation:.2%}")
        if abs(rel_deviation_price_estimation) > self.get_decision_threshold():
            if rel_deviation_price_estimation > 0:
                # print(f"Trader {self.unique_id} is considering buying shares based on price estimation...")
                decision += rel_deviation_price_estimation
            else:
                # print(f"Trader {self.unique_id} is considering selling shares based on price estimation...")
                decision += rel_deviation_price_estimation * -1

        # Example decision logic
        if abs(decision) > self.get_decision_threshold():
            order_size = self.get_order_size()
            if abs(order_size) > 0:
                # print(f"Trader {self.unique_id} is placing an order of size {order_size}...")
                order = LimitOrder(
                    trader_id=self.unique_id,
                    price=self.get_limit_price("buy" if order_size > 0 else "sell"),
                    quantity=order_size,
                    trading_day=self.model.current_day
                )
                # self.placed_orders.append(order)
                self.model.process_order(order)
        # else:
            # print(f"Trader {self.unique_id} decided not to trade.")


    def get_order_size(self):
        diff_portfolio_alloc = self.risk_appetit - self.get_portfolio_share_stocks()
        num_of_stocks = round(diff_portfolio_alloc * self.get_total_wealth() / self.model.get_market_price())
        
        # buy or sell
        is_buy_order = False
        if num_of_stocks > 0:
            is_buy_order = True

        # Ensure the order size is within the available cash
        if is_buy_order and num_of_stocks > self.cash / self.model.get_ask_quote_lob():
            num_of_stocks = round(self.cash / self.model.get_market_price())

        if not is_buy_order and abs(num_of_stocks) > self.num_of_shares:
            num_of_stocks = round(self.num_of_shares)

        min_order_size = 1 if is_buy_order else -1

        if self.risk_appetit <= 0.25:
            num_of_stocks = max(min_order_size, round(num_of_stocks / 5))
        if self.risk_appetit > 0.25 and self.risk_appetit <= 0.5:
            num_of_stocks = max(min_order_size, round(num_of_stocks / 4))
        # e.g. risk appetite 0.5 - 0.7 divided by 4, min order size 1
        if self.risk_appetit > 0.5 and self.risk_appetit <= 0.75:
            num_of_stocks = max(min_order_size, round(num_of_stocks / 3))
        # e.g. risk appetite 0.7 - 0.9 divided by 6, min order size 1
        elif self.risk_appetit > 0.75 and self.risk_appetit <= 1:
            num_of_stocks = max(min_order_size, round(num_of_stocks / 2))

        return num_of_stocks


    def get_limit_price(self, order_type):
        """
        Returns the limit price for the order.
        """
        margin_of_safety = self.risk_aversion / 10
        if order_type == "buy":
            return self.estimated_true_value * (1 - margin_of_safety)
        else:
            return self.estimated_true_value * (1 + margin_of_safety)


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



