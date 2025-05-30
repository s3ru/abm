import logging
import random
from typing import List

import mesa
import numpy as np
from helper import getRandomUniform, lognormal, normal
from limit_order import LimitOrder, OrderStatus

logger = logging.getLogger("batch_log")

class Trader(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pnl = 0.0
        self.is_marginal_trader = True if random.random() <= self.model.share_of_marginal_traders else False
        self.init_wealth = round(self.model.wealth_min * (2 if self.is_marginal_trader else 1) * (1 + np.random.pareto(self.model.wealth_alpha)))
       
        # risk aversion
        self.risk_aversion = round(normal(0.5, 0.1), 2)
        self.risk_appetit = round(1 - self.risk_aversion, 2)
        
        self.last_bought_information = None
        self.informed = False
        self.info_budget_constraint = False
        self.num_bought_information = 0

        # starting portfolio allocation
        self.starting_share_alloc_cash = (self.risk_appetit + getRandomUniform(-0.025, 0.025)) * self.init_wealth
        self.initial_num_shares = round(self.starting_share_alloc_cash / self.model.initial_market_price)
        self.num_of_shares = self.initial_num_shares
        self.initial_cash = self.init_wealth - self.num_of_shares * self.model.initial_market_price
        self.cash = self.initial_cash
        self.cash_chg_liquidity_events = 0
        self.orders: List[LimitOrder] = [] # List of LimitOrder objects

        # skill
        self.skill = max(0.01, min(1, round(normal(0.8 if self.is_marginal_trader else 0.5, 0.05), 2)))

        # valuation bias
        self.valuation_bias = round(normal(0.02, 0.01 if self.is_marginal_trader else 0.04), 4)

        # overconfidence
        self.overconfidence = round(normal(0.4 if self.is_marginal_trader else 0.5, 0.1), 2)

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
        if self.model.current_day == self.model.n_days:
            # liquidation day
            return self.cash + self.num_of_shares * self.model.true_price_avg
        
        return self.cash + self.num_of_shares * self.model.get_market_price()
    
    def get_decision_threshold(self):
        # overconfidence lowers decision treshold
        return self.model.decision_threshold + ((0.5 - self.overconfidence) / 0.5) / 10

    def step(self):
        # print(f"Trader {self.unique_id} starting step...")
        if self.is_marginal_trader:
            self.buy_information()
        else:
            self.estimate_true_value()

        self.decide_to_trade()
        # self.receive_risk_free_rate()
        self.calculate_pnl()

        # print(f"Trader {self.unique_id} completed step.")

    def receive_risk_free_rate(self):
        """
        Updates the cash of the trader based on the risk-free rate.
        """
        if self.model.current_day % 30 == 0:
            interest = self.cash * (self.model.risk_free_rate / 30)
            self.cash += interest

            # Update the cash change due to liquidity events
            self.cash_chg_liquidity_events += interest

        # Calculate the interest earned on the cash balance
        

    def estimate_true_value(self):

        # non marginal traders try to estimate true value based on available market data
        if self.estimated_true_value > 150 or self.estimated_true_value < 50:
            logger.warning(f"Trader {self.unique_id} estimated true value: {self.estimated_true_value}")

        market_price = self.model.get_market_price()
        new_estimation = market_price * (1 + self.valuation_bias)
        noise_abs = market_price * self.model.noise_range
            # basic noise for not highly skilled traders

        # noise trader
        chg = getRandomUniform(-noise_abs, noise_abs)
        new_estimation = round(max(0.01, new_estimation + chg, 2))
        # logger.info(f"Trader {self.unique_id} is adjusting estimation based on noise... {round(chg, 2)}, too {new_estimation}")
        
        if(self.skill > 0.4 and self.skill <= 0.5):
            # look at lob imbalance
            imbalance = self.model.calc_lob_imbalance()
            if imbalance != 0 and abs(imbalance) > 0.1:
                # if more people are on the buy side, the imbalance is positive
                # if more people are on the sell side, the imbalance is negative
                # agent follows the imbalance of the order book -> trend following
                chg = imbalance * getRandomUniform(0.5, 1)
                new_estimation = round(new_estimation + chg, 2)
                # logger.info(f"Trader {self.unique_id} is adjusting estimation based on lob imbalance... {round(chg, 2)}, too {new_estimation}")
        
        if(self.skill > 0.5):
            # look at lob imbalance and vpin
            vpin = self.model.calc_vpin()
            imbalance = self.model.calc_lob_imbalance()

            if vpin > 0.5 and imbalance != 0:
                # informed trade is likely to be present
                if abs(imbalance) > 0.10:
                    # if the imbalance is positive, the trader is more likely to buy
                    chg = imbalance * getRandomUniform(1, 2)
                    new_estimation = round(new_estimation + chg, 2)
            

        self.estimated_true_value = round(new_estimation, 2)



        # print(f"Trader {self.unique_id} estimated true value: {self.estimated_true_value}")


    def buy_information(self):
        if not self.is_marginal_trader:
            # only marginal traders buy information
            return
        
        if random.random() > self.overconfidence:
            # if the trader is overconfident, buys information more frequently
            # also models coordination problem across informed traders
            return
        
        ic = self.model.cost_of_information
        if self.cash < ic or ic > 0.05 * self.get_total_wealth():
            # if the cost of information is too high, the agent won't buy it
            self.info_budget_constraint = True
            return
        else:
            self.info_budget_constraint = False

              
        if self.last_bought_information is not None and (self.model.current_day - self.last_bought_information) < 5:
            # if the trader has already recently bought information, don't buy again
            # simulates coordination problem across informed traders
            self.informed = True
            return
            
        self.informed = True
        self.num_bought_information += 1
        self.last_bought_information = self.model.current_day
        self.cash -= ic
        self.estimated_true_value = round(self.model.true_value * (1 + self.valuation_bias), 2)

    def calculate_pnl(self):
        """
        Calculates the profit and loss (PnL) of the trader.
        """

        is_liqudation_day = self.model.current_day == self.model.n_days
        calculation_price = self.model.true_price_avg if is_liqudation_day else self.model.get_market_price()

        initial_wealth = self.initial_num_shares * calculation_price + self.initial_cash
        current_wealth = self.num_of_shares * calculation_price + self.cash
        current_wealth_adj = current_wealth - self.cash_chg_liquidity_events
        self.pnl = round((current_wealth_adj - initial_wealth) / initial_wealth, 6)
        # if is_liqudation_day:
        #     logger.info(f"Trader {self.unique_id} PnL: {self.pnl:.4%} (liquidation day)")


      # PnL as a percentage of initial wealth
        # print(f"Trader {self.unique_id} PnL: {self.pnl:.2%}")

    def get_portfolio_share_stocks(self):
        """
        Returns the portfolio share of stocks.
        """
        wealth = self.get_total_wealth()
        if wealth == 0:
            logger.warning(f"Trader {self.unique_id} has zero wealth?")
            return self.risk_appetit
        return (self.num_of_shares * self.model.get_market_price()) / wealth
    
    def get_portfolio_share_stocks_with_open_orders(self):

        wealth = self.get_total_wealth()
        if wealth == 0:
            logger.warning(f"Trader {self.unique_id} has zero wealth?")
            return self.risk_appetit

        return ((self.num_of_shares + self.volume_active_orders()) * self.model.get_market_price()) / wealth

    def has_active_orders(self):
        """
        Returns True if the trader has active orders, False otherwise.
        """
        return any(order.get_status() == OrderStatus.OPEN or order.get_status() == OrderStatus.PARTIALLY_FILLED for order in self.orders)

    def volume_active_orders(self):
        """
        Returns the volume of active orders.
        """
        return sum(order.get_quantity_unfilled() for order in self.orders if order.get_status() == OrderStatus.OPEN or order.get_status() == OrderStatus.PARTIALLY_FILLED)

    def decide_to_trade(self):

        if not self.informed and random.random() > self.overconfidence:
            # if informed do trade
            return
        
        decision = 0

        # decide to trade if portfolio allocation is not in line with risk appetite
        portfolio_share_stocks = self.get_portfolio_share_stocks_with_open_orders()
        diff_portfolio_alloc = self.risk_appetit - portfolio_share_stocks

        if abs(diff_portfolio_alloc) > 0.1 and not self.is_marginal_trader:
            # if the difference is greater than 10%, the trader decides to trade
            decision += diff_portfolio_alloc
        

        # decide to trade if own price estimation is significantly different from market price
        rel_deviation_price_estimation = (self.estimated_true_value - self.model.get_market_price()) / self.model.get_market_price()
        decision += rel_deviation_price_estimation
            
        threshold = self.get_decision_threshold()
        if abs(decision) > threshold:
            # apply threshold to avoid excessive trading
            direction = "buy" if decision > 0 else "sell"
            order_size = self.get_order_size(direction)
            if abs(order_size) > 0:
                # print(f"Trader {self.unique_id} is placing a {direction} order of size {order_size}...")
                order = LimitOrder(
                    trader_id=self.unique_id,
                    price=self.get_limit_price(direction),
                    quantity=order_size,
                    trading_day=self.model.current_day,
                    is_trader_mt=self.is_marginal_trader
                )
                self.orders.append(order)
                self.model.process_order(order)
                if self.is_marginal_trader:
                    logger.info(f"{self.unique_id}: MT placed a {direction} order of size {order_size} at limit price {order.limit_price}, expected price is {self.estimated_true_value} Bias({self.valuation_bias:.2%}), true value is {self.model.true_value}, market price is {self.model.get_market_price()}")
        # else:
            # print(f"Trader {self.unique_id} decided not to trade.")

    def get_stocks_sold_volume(self):
        """
        Returns the volume of stocks sold.
        """
        return sum(abs(order.quantity) - abs(order.get_quantity_unfilled()) for order in self.orders if order.get_order_type() == "sell" and (order.get_status() == OrderStatus.FILLED or order.get_status() == OrderStatus.PARTIALLY_FILLED))


    def get_avg_price_sold_volume(self):
        """
        Returns the average price of stocks sold.
        """
        if self.get_stocks_sold_volume() == 0:
            return 0

        return sum(order.get_avg_execution_price() * (order.quantity + order.get_quantity_unfilled()) for order in self.orders if order.get_order_type() == "sell" and (order.get_status() == OrderStatus.FILLED or order.get_status() == OrderStatus.PARTIALLY_FILLED)) / self.get_stocks_sold_volume()

    def get_stocks_bought_volume(self):
        """
        Returns the volume of stocks bought.
        """
        return sum(order.quantity - order.get_quantity_unfilled() for order in self.orders if order.get_order_type() == "buy" and (order.get_status() == OrderStatus.FILLED or order.get_status() == OrderStatus.PARTIALLY_FILLED))


    def get_avg_price_bought_volume(self):
        """
        Returns the average price of stocks bought.
        """
        if self.get_stocks_bought_volume() == 0:
            return 0

        return sum(order.get_avg_execution_price() * (order.quantity - order.get_quantity_unfilled()) for order in self.orders if order.get_order_type() == "buy" and (order.get_status() == OrderStatus.FILLED or order.get_status() == OrderStatus.PARTIALLY_FILLED)) / self.get_stocks_bought_volume()




    def get_order_size(self, direction):
              
        price = 0
        if direction == 'buy':
            price = self.model.get_ask_quote_lob()
        else:
            price = self.model.get_bid_quote_lob()

        max_risk_wealth = self.get_total_wealth() * 0.05
        max_num_of_stocks = max_risk_wealth / price
        if direction == "buy":
            # Ensure the order size is within the available cash
            max_stocks_cash = round(self.cash / price)
            num_of_stocks = min(max_num_of_stocks, max_stocks_cash)
        else:
            # Ensure agent owns sold stocks (no short-sellling)
            own_num_of_stocks = round(self.num_of_shares)
            num_of_stocks = min(own_num_of_stocks, max_num_of_stocks)
       
        # Adjust the order size based on risk appetite
        if self.risk_appetit < 0.5:
            num_of_stocks = max(1, round(num_of_stocks / 2))


        return abs(round(num_of_stocks)) * (1 if direction == "buy" else -1)  # Return positive for buy, negative for sell


    def get_limit_price(self, order_type):
        """
        Returns the limit price for the order.
        """

        return round(self.estimated_true_value, 2)
    
        # # Example logic for determining the limit price
        # safety_margin = self.estimated_true_value * 0.02
        # if order_type == "buy":
        #     return round(self.estimated_true_value - self.risk_aversion * safety_margin, 2)
        # else:
        #     return round(self.estimated_true_value + self.risk_aversion * safety_margin, 2)


    



