
from datetime import datetime, timedelta
import logging
import mesa
from typing import Dict, List
from helper import lognormal, normal, poisson

from mesa.datacollection import DataCollector
import numpy as np
import random

from limit_order import LimitOrder, OrderStatus, Transaction
from trader import Trader
import copy

logger = logging.getLogger("batch_log")


class LimitOrderMarket(mesa.Model):
    def __init__(
            self,
            num_agents: int = 100,
            share_of_marginal_traders: float = 0.5,
            n_days: int = 100,
            cost_of_information: int = 500,
            risk_free_rate: float = 0.02,
            start_true_value = 100.0,
            decision_threshold = 0.05,
            order_expiration = 5,
            noise_range = 0.03,
            wealth_min=5000,
            event_info_frequency = 0.1,
            event_info_intensity = 0.25,
            event_liquidity_frequency = 0.01,
            event_liquidity_intensity = 100,
            # starting_phase = 5,
        ) -> None:
        super().__init__(seed=None)
        # wealth
        self.wealth_alpha = 2.0       # shape
        self.wealth_min = wealth_min       # minimum wealth

        self.risk_free_rate = risk_free_rate

        self.cost_of_information = cost_of_information
        self.share_of_marginal_traders = share_of_marginal_traders

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
        self.true_value: float = start_true_value
        self.initial_market_price: float = round(normal(self.true_value, 1), 2) # initial market price
        # print(f"Initial market price: {self.initial_market_price}")

        self.volume_mt_involment = 0
        self.volume = 0
        self.high_price = 0
        self.low_price = 0
        self.open_price = 0
        self.close_price = 0

        self.vpin = 0
        self.lob_imbalance = 0

        self.start_date = datetime.today()

        self.lob: List[LimitOrder] = [] # List of LimitOrder objects
        self.transactions: List[Transaction] = [] # List of transactions (price, volume)
        self.information_events: List[bool] = []  # List of information events
        self.last_market_price: float = self.initial_market_price
        self.true_prices: List[float] = []
        self.true_price_avg: float = 0
        self.info_event_occurred: bool = False
        self.datacollector = DataCollector(
            model_reporters={
                             "trading_day": lambda m: m.current_day,
                             "trading_date": lambda x: x.start_date + timedelta(days=x.current_day),
                             "market_price": lambda m: m.get_market_price(),
                             "info_event":  "info_event_occurred",
                             "true_value": lambda m: m.true_value,
                             "bid_ask_spread": lambda m: m.get_bid_ask_spread(),
                             "volume": "volume",
                             "volume_mt_involment": "volume_mt_involment",
                             "open_price": "open_price",
                             "high_price": "high_price",
                             "low_price": "low_price",
                             "vpin": "vpin",
                             "buy_orders": lambda m: m.get_number_of_buy_orders(),
                             "sell_orders": lambda m: m.get_number_of_sell_orders(),
                             "close_price": "close_price",
                             "shares_outstanding": lambda m: m.get_total_shares_outstanding(),
                             "lob_imbalance": "lob_imbalance",
                             "cost_of_information": "cost_of_information",
                             "share_of_marginal_traders": "share_of_marginal_traders",
                             "rel_distance_sell_orders": lambda m: m.get_rel_distance_sell_orders(),
                             "rel_distance_buy_orders": lambda m: m.get_rel_distance_buy_orders(),
                             },
            agent_reporters={
                "PnL": "pnl",
                "wealth": lambda t: t.get_total_wealth(),
                "is_marginal_trader": "is_marginal_trader",
                "cash": "cash",
                "skill": "skill",
                "num_of_shares": "num_of_shares",
                "informed": "informed",
                "stocks_sold": lambda t: t.get_stocks_sold_volume(),
                "stocks_sold_avg_price": lambda t: t.get_avg_price_sold_volume(),
                "stocks_bought": lambda t: t.get_stocks_bought_volume(),
                "stocks_bought_avg_price": lambda t: t.get_avg_price_bought_volume(),
                "cash_chg_liquidity_events": "cash_chg_liquidity_events",	
                "num_bought_information": "num_bought_information",
                "info_budget_constraint": "info_budget_constraint",
                }
        )

        # print(f"""LOM 
        #       ================================
        #       initial market price: {self.initial_market_price},
        #       true_value: {self.true_value}, 
        #       number of agents: {self.num_agents}, 
        #       number of days: {self.n_days}""")
      

        Trader.create_agents(model=self, n=num_agents)


    def step(self) -> None:
        logger.info(f"Starting step {self.current_day}... {self.share_of_marginal_traders:.0%} MT, {self.cost_of_information} IC")
        logger.info(f"Number of sell orders in lob before processing: {len(self.get_sell_orders())}")
        logger.info(f"Number of buy orders in lob before processing: {len(self.get_buy_orders())}")
        # print(f"Market price: {self.get_market_price()}")

        self.set_orders_expired()
        self.markov_price_step()

        # Fundamentalwert aktualisieren
        if poisson(self.event_info_frequency):
            self.info_event_occurred = True
            old_price = self.true_value
            shock = lognormal(1, self.event_info_intensity)
            # print(f"Shock applied: {shock}")
            shock_with_direction =  round(shock if random.random() < 0.5 else -1 * shock)
            self.true_value = max(1, self.true_value + shock_with_direction)  # ensure price is positive
            # print(f"Shock applied: trading_day={self.current_day}, old_price={old_price}, new_price={self.true_value}, diff={(self.true_value - old_price)/old_price:.2%}")
        else:
            self.info_event_occurred = False

        # Liquiditätsereignisse
        for agent in self.agents:
            if poisson(self.event_liquidity_frequency):
                old_cash = agent.cash
                cash_change = round(normal(0.01 * agent.init_wealth, self.event_liquidity_intensity))
                cash_change_abs = cash_change * random.choice([-1, 1])
                agent.cash += cash_change_abs
                agent.cash_chg_liquidity_events += cash_change_abs
                # print(f"Agent {agent.unique_id} cash updated by {cash_change}. New cash: {agent.cash}, diff= {(agent.cash - old_cash)/old_cash:.2%}")

        self.agents.shuffle_do("step")

        self.set_open_high_low_closing_price_trading_day()
        self.calc_lob_imbalance()
        self.calc_vpin()
        self.true_prices.append(self.true_value)
        self.true_price_avg = np.mean(self.true_prices)
        self.datacollector.collect(self)
        self.current_day += 1
        # print(f"Step {self.current_day} completed.")

    def get_total_shares_outstanding(self):
        """
        Returns the total number of shares outstanding in the market.
        """
        total_shares = sum(agent.num_of_shares for agent in self.agents)
        return total_shares
    
    def set_open_high_low_closing_price_trading_day(self):
        """
        Sets the open, high, low, and closing prices for the current trading day.
        """
        if self.current_day == 0:
            self.open_price = self.initial_market_price
        else:
            self.open_price = self.close_price

        if len(self.transactions) > 0:
            prices = [transaction.price for transaction in self.transactions if transaction.trading_day == self.current_day]
            if len(prices) > 0:
                self.high_price = max(prices)
                self.low_price = min(prices)
                self.close_price = prices[-1]
                self.volume = self.get_volume_current_trading_day()
                self.volume_mt_involment = sum(transaction.volume for transaction in self.transactions if transaction.mt_involvement and transaction.trading_day == self.current_day)
            else:
                self.high_price = self.open_price
                self.low_price = self.open_price
                self.close_price = self.open_price
                self.volume = 0
                self.volume_mt_involment = 0
        else:
            self.high_price = self.open_price
            self.low_price = self.open_price
            self.close_price = self.open_price
            self.volume = 0
            self.volume_mt_involment = 0

    def get_market_price(self) -> float:
        if len(self.transactions) > 0:
            last_transaction = self.transactions[-1]
            self.last_market_price = last_transaction.price
        elif len(self.get_sell_orders()) > 0 or len(self.get_buy_orders()) > 0:
            self.last_market_price = (self.get_ask_quote_lob() + self.get_bid_quote_lob()) / 2
        
        # diff = abs(self.last_market_price - self.true_value)
        # if diff > 50:
        #     print(f"Market price is too high: {self.last_market_price}. Setting to 100.")

        return self.last_market_price
        
    def calc_lob_imbalance(self) -> float: 
        """
        Returns the relative depth of the limit order book (LOB).
        """

        self.lob_imbalance = 0.0

        buy_orders = self.get_buy_orders()
        sell_orders = self.get_sell_orders()
        if len(buy_orders) > 0 and len(sell_orders) > 0:
            volume_buy_orders = sum(order.get_quantity_unfilled() for order in buy_orders)
            volume_sell_orders = sum(order.get_quantity_unfilled() for order in sell_orders)
            volume_combined = volume_buy_orders + abs(volume_sell_orders)
            self.lob_imbalance = (volume_buy_orders + volume_sell_orders) / volume_combined

        return self.lob_imbalance

    def get_rel_distance_sell_orders(self):
        """
        Returns the relative distance of the limit order book (LOB) from the market price.
        """
        sell_orders = self.get_sell_orders()
        if len(sell_orders) > 0:
            median_sell_order_price = np.median([order.limit_price for order in sell_orders])
            ask = self.get_ask_quote_lob()
            rel_distance = abs(ask - median_sell_order_price) / ask
            return rel_distance
        else:
            return 0.0
        
    def get_number_of_buy_orders(self) -> int:
        """
        Returns the number of buy orders in the limit order book (LOB).
        """
        buy_orders = self.get_buy_orders()
        return len(buy_orders)
    
    def get_number_of_sell_orders(self) -> int:
        """
        Returns the number of sell orders in the limit order book (LOB).
        """
        sell_orders = self.get_sell_orders()
        return len(sell_orders) 

    def calc_vpin(self):
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)
        """

        self.vpin = 0.0

        transaktions = [transaction for transaction in self.transactions if transaction.trading_day == self.current_day-1 or transaction.trading_day == self.current_day]
        if len(transaktions) > 50:
            volume_buy_initiated = sum(transaction.volume for transaction in transaktions if transaction.get_initiator() == "buy")
            volume_sell_initiated = sum(transaction.volume for transaction in transaktions if transaction.get_initiator() == "sell")

            volume_combined = volume_buy_initiated + volume_sell_initiated

            self.vpin = abs(volume_buy_initiated - volume_sell_initiated) / volume_combined
        
        return self.vpin
        
    def get_rel_distance_buy_orders(self):
        """
        Returns the relative distance of the limit order book (LOB) from the market price.
        """
        buy_orders = self.get_buy_orders()
        if len(buy_orders) > 0:
            median_buy_order_price = np.median([order.limit_price for order in buy_orders])
            bid = self.get_bid_quote_lob()
            rel_distance = abs(bid - median_buy_order_price) / bid
            return rel_distance
        else:
            return 0.0

    def set_orders_expired(self):
        for order in self.lob:
            if order.get_status() == OrderStatus.OPEN or order.get_status() and (self.current_day - order.trading_day) > self.order_expiration:
                if random.random() > 0.5:
                    order.is_canceled = True
                # print(f"Order {order.order_id} expired and canceled.")
                self.lob.remove(order)

    def process_order(self, order: LimitOrder) -> None:
        """
        Accepts a limit order, (partially) executes it and adds it to the limit order book (LOB).
        """
        if order.get_order_type() == "buy":
            buy_order = order
            sell_orders = self.get_sell_orders()
            if len(sell_orders) > 0:
                for sell_order in sell_orders:
                    if buy_order.get_quantity_unfilled() == 0:
                        break

                    if sell_order.limit_price <= buy_order.limit_price:
                        # Execute order as market order
                        transaction = Transaction(
                            price=sell_order.limit_price,
                            volume=min(abs(order.get_quantity_unfilled()), abs(sell_order.get_quantity_unfilled())),
                            buyer_order= copy.copy(buy_order),
                            seller_order= copy.copy(sell_order),
                            trading_day=self.current_day
                        )
                        self.transactions.append(transaction)
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
                # Add the order to the LOB if unfilled quantity left
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
            # # nobody wants to buy decreasing price
            # if self.lob_imbalance != 0 and self.lob_imbalance < -0.5:
            #     # print(f"Nobody wants to buy, decreasing price")
            #     return self.last_market_price * (1 - 0.01) 
            # else:
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
            # if self.lob_imbalance != 0 and self.lob_imbalance > 0.5:
            #     # print(f"Nobody wants to sell, increasing price")
            #     return self.last_market_price * (1 + 0.01)
            # else:
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
        phi = 0.9
        sigma = 0.7
        # old_price = self.true_value
        self.true_value = max(1, round(mu + phi * (self.true_value - mu) + np.random.normal(0, sigma), 2))
        # print(f"Markov price step: old_price={old_price}, new_price={self.true_value}, diff={(self.true_value - old_price)/old_price:.4%}")

    def run_model(self):
        for _ in range(self.n_days):
            self.step()

        self.evaluate_efficiency()
