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
from typing import Dict, List
from helper import lognormal, normal, poisson

from mesa.datacollection import DataCollector
import numpy as np
import random
import math

from limit_order import LimitOrder, OrderStatus, Transaction
from trader import Trader
import copy

class LimitOrderMarket(mesa.Model):
    def __init__(self, num_agents: int = 100, n_days: int = 100, decision_threshold = 0.05, order_expiration = 10) -> None:
        super().__init__(seed=None)
        # wealth
        self.wealth_alpha = 2.0       # shape
        self.wealth_min = 1000.0       # minimum wealth

        self.num_agents: int = num_agents
        self.current_day: int = 0
        self.decision_threshold: float = decision_threshold
        self.order_expiration: int = order_expiration
        self.n_days: int = n_days
        self.current_v: float = 100.0
        self.initial_market_price: float = round(normal(self.current_v, 40))
        self.lob: List[LimitOrder] = [] # List of LimitOrder objects
        self.transactions: List[Transaction] = [] # List of transactions (price, volume)
        self.volumes: Dict[int, float] = {}  # List of volumes
        self.information_events: List[bool] = []  # List of information events
        self.data_collector = DataCollector(
            model_reporters={"market_price": lambda m: m.get_market_price(),
                             "true_value": lambda m: m.current_v, 
                             "volume": lambda m: m.get_volume_current_trading_day()},
            agent_reporters={"PnL": "pnl", "wealth": lambda t: t.get_total_wealth(), "cash": "cash", "num_of_shares": "num_of_shares",}
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
        self.set_orders_expired()
        self.markov_price_step()

        # Fundamentalwert aktualisieren
        if poisson(0.2):
            self.information_events.append(True)
            old_price = self.current_v
            shock = lognormal(0, 0.2)
            # print(f"Shock applied: {shock}")
            self.current_v += round(shock if random.random() < 0.5 else -1 * shock)
            print(f"Shock applied: old_price={old_price}, new_price={self.current_v}, diff={(self.current_v - old_price)/old_price:.2%}")
        else:
            self.information_events.append(False)

        # Liquiditätsereignisse
        for agent in self.agents:
            if poisson(0.05):
                cash_change = round(normal(100, 10))
                agent.cash += cash_change * random.choice([-1, 1])
                print(f"Agent {agent.unique_id} cash updated by {cash_change}. New cash: {agent.cash}")

        self.agents.shuffle_do("step")

        self.data_collector.collect(self)

        self.current_day += 1
        print(f"Step {self.current_day} completed.")

    def get_market_price(self) -> float:
        if len(self.get_sell_orders()) > 0 and len(self.get_buy_orders()) > 0:
            return self.get_ask_quote_lob().price + self.get_bid_quote_lob().price / 2
        else:
            return self.initial_market_price

    def evaluate_efficiency(self):
        prices = self.data_collector.get_model_vars_dataframe()["market_price"].values
        true_vals = self.data_collector.get_model_vars_dataframe()["true_value"].values
        avg_deviation = np.mean(np.abs(prices - true_vals))
        print(f"Durchschnittliche Abweichung Marktpreis vs. fundamentaler Wert: {avg_deviation:.2f}")

    def set_orders_expired(self):
        for order in self.lob:
            if order.get_status() == OrderStatus.OPEN or order.get_status() and (self.current_day - order.trading_day) > self.order_expiration:
                order.is_canceled = True
                print(f"Order {order.order_id} expired and canceled.")
                self.lob.remove(order)

    def process_order(self, order: LimitOrder) -> None:
        """
        Accepts a limit order and adds it to the limit order book (LOB).
        """
        if order.get_order_type() == "buy":
            buy_order = order
            sell_orders = self.get_sell_orders()
            if len(sell_orders) > 0:
                for sell_order in sell_orders:
                    if buy_order.get_quantity_unfilled() == 0:
                        break

                    if sell_order.limit_price < buy_order.limit_price:
                        # Execute transaction
                        transaction = Transaction(
                            price=sell_order.limit_price,
                            volume=min(order.get_quantity_unfilled(), sell_order.get_quantity_unfilled()),
                            buyer_order= copy.copy(buy_order),
                            seller_order= copy.copy(sell_order),
                            trading_day=self.current_day
                        )
                        self.transactions.append(transaction)
                        print(f"Transaction executed: {transaction}")
                        sell_order.transactions.append(transaction)
                        buy_order.transactions.append(transaction)

                        buyer = self.agents.get(buy_order.trader_id, None)
                        buyer.num_of_shares += transaction.volume
                        buyer.cash -= transaction.volume * transaction.price

                        seller = self.agents.get(sell_order.trader_id, None)
                        seller.num_of_shares -= transaction.volume
                        seller.cash += transaction.volume * transaction.price
       
                        if sell_order.quantity == 0:
                            self.lob.remove(sell_order)

            if buy_order.get_quantity_unfilled() > 0:
                # Add the order to the LOB
                self.lob.append(buy_order)

        elif order.get_order_type() == "sell":
            sell_order = order
            buy_orders = self.get_buy_orders()
            if len(buy_orders) > 0:
                for buy_order in buy_orders:
                    if sell_order.get_quantity_unfilled() == 0:
                        break

                    if buy_order.limit_price > sell_order.limit_price:
                        # Execute transaction
                        transaction = Transaction(
                            price=buy_order.limit_price,
                            volume=min(order.get_quantity_unfilled(), buy_order.get_quantity_unfilled()),
                            buyer_order= copy.copy(buy_order),
                            seller_order= copy.copy(sell_order),
                            trading_day=self.current_day
                        )
                        self.transactions.append(transaction)
                        print(f"Transaction executed: {transaction}")
                        buy_order.transactions.append(transaction)
                        sell_order.transactions.append(transaction)

                        buyer = self.agents.get(buy_order.trader_id, None)
                        buyer.num_of_shares += transaction.volume
                        buyer.cash -= transaction.volume * transaction.price

                        seller = self.agents.get(sell_order.trader_id, None)
                        seller.num_of_shares -= transaction.volume
                        seller.cash += transaction.volume * transaction.price


                        if buy_order.quantity == 0:
                            self.lob.remove(buy_order)

        
            return
            

        self.lob.append(order)
        print(f"Order accepted: {order}")

    def get_volume_current_trading_day(self) -> float:
        self.transactions_today = [transaction for transaction in self.transactions if transaction.trading_day == self.current_day]
        return sum(transaction.volume for transaction in self.transactions_today)

    def get_buy_orders(self):
        """
        Returns all buy orders from the limit order book (LOB).
        """
        buy_orders = [order for order in self.lob if order.get_order_type() == "buy"]
        buy_orders.sort(key=lambda x: (-x.price, x.timestamp))
        # print(f"Buy orders in LOB: {buy_orders}")
        return buy_orders
    
    def get_sell_orders(self):
        """
        Returns all sell orders from the limit order book (LOB).
        """
        sell_orders = [order for order in self.lob if order.get_order_type() == "sell"]
        sell_orders.sort(key=lambda x: (x.price, x.timestamp))
        # print(f"Sell orders in LOB: {sell_orders}")
        return sell_orders

    def get_bid_quote_lob(self): 
        """
        Returns the best bid quote from the limit order book (LOB).
        """
        buy_orders = self.get_buy_orders()
        if len(buy_orders) > 0:
            # Sort orders by price and then by timestamp
            best_bid = self.lob[0]
            # print(f"Best bid in LOB: {best_bid}")
            return best_bid
        else:
            return self.get_market_price() - 0.50
        
    
    
    
    def get_ask_quote_lob(self): 
        """
        Returns the best ask quote from the limit order book (LOB).
        """
        sell_orders = self.get_sell_orders()
        if len(sell_orders) > 0:
            # Sort orders by price and then by timestamp
            
            best_ask = self.lob[0]
            # print(f"Best ask in LOB: {best_ask}")
            return best_ask.price
        else:
            return self.get_market_price() + 0.50

    def markov_price_step(self):
        mu = 100
        phi = 0.95
        sigma = 1.0
        old_price = self.current_v
        self.current_v = round(mu + phi * (self.current_v - mu) + np.random.normal(0, sigma), 2)
        print(f"Markov price step: old_price={old_price}, new_price={self.current_v}, diff={(self.current_v - old_price)/old_price:.4%}")

    def run_model(self):
        for _ in range(self.n_days):
            self.step()

        self.evaluate_efficiency()
