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

from limit_order import LimitOrder, OrderStatus, Transaction
from trader import Trader
import copy

from viz import create_viz

class LimitOrderMarket(mesa.Model):
    def __init__(
            self,
            num_agents: int = 100,
            n_days: int = 100,
            start_true_value = 100.0,
            decision_threshold = 0.05,
            order_expiration = 10,
            noise_range = 0.03,
            wealth_min=5000,
            event_info_frequency = 0.2,
            event_info_intensity = 0.5,
            event_liquidity_frequency = 0.05,
            event_liquidity_intensity = 1000,
            # starting_phase = 5,
        ) -> None:
        super().__init__(seed=None)
        # wealth
        self.wealth_alpha = 2.0       # shape
        self.wealth_min = wealth_min       # minimum wealth

        # init event parameters
        self.event_info_frequency = event_info_frequency
        self.event_info_intensity = event_info_intensity
        self.event_liquidity_frequency = event_liquidity_frequency
        self.event_liquidity_intensity = event_liquidity_intensity

        self.noise_range: float = noise_range 
        self.num_agents: int = num_agents
        self.current_day: int = 0
        self.decision_threshold: float = decision_threshold
        self.order_expiration: int = order_expiration
        self.n_days: int = n_days
        # self.starting_phase: int = starting_phase
        self.current_v: float = start_true_value
        self.initial_market_price: float = round(normal(self.current_v, 5))
        # print(f"Initial market price: {self.initial_market_price}")

        self.lob: List[LimitOrder] = [] # List of LimitOrder objects
        self.transactions: List[Transaction] = [] # List of transactions (price, volume)
        self.volumes: Dict[int, float] = {}  # List of volumes
        self.information_events: List[bool] = []  # List of information events
        self.last_market_price: float = self.initial_market_price
        self.info_event_occurred: bool = False
        self.datacollector = DataCollector(
            model_reporters={
                             "trading_day": lambda m: m.current_day,
                             "market_price": lambda m: m.get_market_price(),
                             "info_event":  lambda m: m.info_event_occurred,
                             "true_value": lambda m: m.current_v,
                             "bid_ask_spread": lambda m: m.get_bid_ask_spread(),
                             "volume": lambda m: m.get_volume_current_trading_day()},
            agent_reporters={"PnL": "pnl", "wealth": lambda t: t.get_total_wealth(), "cash": "cash", "num_of_shares": "num_of_shares",}
        )

        # print(f"""LOM 
        #       ================================
        #       initial market price: {self.initial_market_price},
        #       current_v: {self.current_v}, 
        #       number of agents: {self.num_agents}, 
        #       number of days: {self.n_days}""")
      

        Trader.create_agents(model=self, n=num_agents)


    def step(self) -> None:
        # print(f"Starting step {self.current_day}...")
        # print(f"Number of sell orders in lob before processing: {len(self.get_sell_orders())}")
        # print(f"Number of buy orders in lob before processing: {len(self.get_buy_orders())}")
        # print(f"Market price: {self.get_market_price()}")

        self.set_orders_expired()
        self.markov_price_step()

        # Fundamentalwert aktualisieren
        if poisson(self.event_info_frequency):
            self.info_event_occurred = True
            old_price = self.current_v
            shock = lognormal(1, self.event_info_intensity)
            # print(f"Shock applied: {shock}")
            shock_with_direction =  round(shock if random.random() < 0.5 else -1 * shock)
            self.current_v = max(1, self.current_v + shock_with_direction)  # ensure price is positive
            # print(f"Shock applied: trading_day={self.current_day}, old_price={old_price}, new_price={self.current_v}, diff={(self.current_v - old_price)/old_price:.2%}")
        else:
            self.info_event_occurred = False

        # Liquiditätsereignisse
        for agent in self.agents:
            if poisson(self.event_liquidity_frequency):
                old_cash = agent.cash
                cash_change = round(normal(0.05 * agent.init_wealth, self.event_liquidity_intensity))
                agent.cash += cash_change * random.choice([-1, 1])
                # print(f"Agent {agent.unique_id} cash updated by {cash_change}. New cash: {agent.cash}, diff= {(agent.cash - old_cash)/old_cash:.2%}")

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

        self.current_day += 1
        # print(f"Step {self.current_day} completed.")

    def get_market_price(self) -> float:
        if len(self.transactions) > 0:
            last_transaction = self.transactions[-1]
            self.last_market_price = last_transaction.price
        elif len(self.get_sell_orders()) > 0 or len(self.get_buy_orders()) > 0:
            self.last_market_price = (self.get_ask_quote_lob() + self.get_bid_quote_lob()) / 2
        
        return self.last_market_price
        
    def get_lob_tilt(self) -> float: 
        """
        Returns the relative depth of the limit order book (LOB).
        """
        buy_orders = self.get_buy_orders()
        sell_orders = self.get_sell_orders()
        if len(buy_orders) > 0 or len(sell_orders) > 0:
            volume_buy_orders = sum(order.get_quantity_unfilled() for order in buy_orders)
            volume_sell_orders = sum(order.get_quantity_unfilled() for order in sell_orders)
            volume_combined = volume_buy_orders + abs(volume_sell_orders)
            lob_tilt = (volume_buy_orders + volume_sell_orders) / volume_combined
            return lob_tilt
        else:
            return 0.0


    def set_orders_expired(self):
        for order in self.lob:
            if order.get_status() == OrderStatus.OPEN or order.get_status() and (self.current_day - order.trading_day) > self.order_expiration:
                order.is_canceled = True
                # print(f"Order {order.order_id} expired and canceled.")
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

                    if sell_order.limit_price <= buy_order.limit_price:
                        # Execute transaction
                        transaction = Transaction(
                            price=sell_order.limit_price,
                            volume=min(abs(order.get_quantity_unfilled()), abs(sell_order.get_quantity_unfilled())),
                            buyer_order= copy.copy(buy_order),
                            seller_order= copy.copy(sell_order),
                            trading_day=self.current_day
                        )
                        self.transactions.append(transaction)
                        # print(f"Transaction executed: {transaction}")
                        sell_order.transactions.append(transaction)
                        buy_order.transactions.append(transaction)

                        buyer =  self.find_trader_by_id(buy_order.trader_id)
                        buyer.num_of_shares += transaction.volume
                        buyer.cash -= transaction.volume * transaction.price

                        seller = self.find_trader_by_id(sell_order.trader_id)
                        seller.num_of_shares -= transaction.volume
                        seller.cash += transaction.volume * transaction.price
       
                        if sell_order.get_quantity_unfilled() == 0:
                            self.lob.remove(sell_order)

            if abs(buy_order.get_quantity_unfilled()) > 0:
                # Add the order to the LOB
                #print(f"Adding buy order to LOB: {buy_order}")
                self.lob.append(buy_order)

        elif order.get_order_type() == "sell":
            sell_order = order
            buy_orders = self.get_buy_orders()
            if len(buy_orders) > 0:
                for buy_order in buy_orders:
                    if sell_order.get_quantity_unfilled() == 0:
                        break

                    if buy_order.limit_price >= sell_order.limit_price:
                        # Execute transaction
                        transaction = Transaction(
                            price=buy_order.limit_price,
                            volume=min(abs(order.get_quantity_unfilled()), abs(buy_order.get_quantity_unfilled())),
                            buyer_order= copy.copy(buy_order),
                            seller_order= copy.copy(sell_order),
                            trading_day=self.current_day
                        )
                        self.transactions.append(transaction)
                        # print(f"Transaction executed: {transaction}")
                        buy_order.transactions.append(transaction)
                        sell_order.transactions.append(transaction)

                        buyer =  self.find_trader_by_id(buy_order.trader_id)
                        buyer.num_of_shares += transaction.volume
                        buyer.cash -= transaction.volume * transaction.price

                        seller = self.find_trader_by_id(sell_order.trader_id)
                        seller.num_of_shares -= transaction.volume
                        seller.cash += transaction.volume * transaction.price


                        if buy_order.get_quantity_unfilled() == 0:
                            self.lob.remove(buy_order)

            if abs(sell_order.get_quantity_unfilled()) > 0:
                # Add the order to the LOB
                # print(f"Adding sell order to LOB: {sell_order}")
                self.lob.append(sell_order)
            return
            
    def find_trader_by_id(self, agent_id: int) -> Trader:
        """
        Finds an agent by its unique ID.
        """
        agent_set = self.agents.select(lambda a: a.unique_id == agent_id, 1)
        return agent_set[0]
        
        

    def get_volume_current_trading_day(self) -> float:
        self.transactions_today = [transaction for transaction in self.transactions if transaction.trading_day == self.current_day]
        return sum(transaction.volume for transaction in self.transactions_today)

    def get_buy_orders(self):
        """
        Returns all buy orders from the limit order book (LOB).
        """
        buy_orders = [order for order in self.lob if order.get_order_type() == "buy" and (order.get_status() == OrderStatus.OPEN or order.get_status() == OrderStatus.PARTIALLY_FILLED)]
        buy_orders.sort(key=lambda x: (-x.limit_price, x.timestamp))
        # print(f"Buy orders in LOB: {buy_orders}")
        return buy_orders
    
    def get_sell_orders(self):
        """
        Returns all sell orders from the limit order book (LOB).
        """
        sell_orders = [order for order in self.lob if order.get_order_type() == "sell" and (order.get_status() == OrderStatus.OPEN or order.get_status() == OrderStatus.PARTIALLY_FILLED)]
        sell_orders.sort(key=lambda x: (x.limit_price, x.timestamp))
        # print(f"Sell orders in LOB: {sell_orders}")
        return sell_orders

    def get_bid_quote_lob(self): 
        """
        Returns the best bid quote from the limit order book (LOB).
        """
        buy_orders = self.get_buy_orders()
        if len(buy_orders) > 0:
            # Sort orders by price and then by timestamp
            best_bid = buy_orders[0]
            # print(f"Best bid in LOB: {best_bid}")
            return best_bid.limit_price
        else:
            # nobody wants to buy decreasing price
            if self.get_lob_tilt() < -0.5:
                # print(f"Nobody wants to buy, decreasing price")
                return self.last_market_price * (1 - 0.01) 
            else:
                return self.last_market_price
        
    
    
    
    def get_ask_quote_lob(self): 
        """
        Returns the best ask quote from the limit order book (LOB).
        """
        sell_orders = self.get_sell_orders()
        if len(sell_orders) > 0:
            # Sort orders by price and then by timestamp
            
            best_ask = sell_orders[0]
            # print(f"Best ask in LOB: {best_ask}")
            return best_ask.limit_price
        else:
            # nobody wants to sell increasing price
            if self.get_lob_tilt() > 0.5:
                # print(f"Nobody wants to sell, increasing price")
                return self.last_market_price * (1 + 0.01)
            else:
                return self.last_market_price
        

    def get_bid_ask_spread(self) -> float:
        """
        Returns the bid-ask spread.
        """
        bid = self.get_bid_quote_lob()
        ask = self.get_ask_quote_lob()
        spread = ask - bid
        # print(f"Bid-ask spread: {spread}")
        return spread

    def markov_price_step(self):
        mu = 100
        phi = 0.8
        sigma = 1
        old_price = self.current_v
        self.current_v = max(1, round(mu + phi * (self.current_v - mu) + np.random.normal(0, sigma), 2))
        # print(f"Markov price step: old_price={old_price}, new_price={self.current_v}, diff={(self.current_v - old_price)/old_price:.4%}")

    def run_model(self):
        for _ in range(self.n_days):
            self.step()

        self.evaluate_efficiency()
