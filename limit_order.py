from typing import List
import uuid
from datetime import datetime

from transaction import Transaction

class LimitOrder:
    def __init__(self, trader_id: int, price: float, quantity: int, trading_day: int):
        """
        Initialize a limit order.

        :param order_id: Unique identifier for the order.
        :param trader_id: ID of the trader placing the order.
        :param order_type: Type of the order ('buy' or 'sell').
        :param price: Price at which the order is placed.
        :param quantity: Quantity of the asset to buy or sell.
        :param timestamp: Time when the order was placed.
        """
        self.order_id = uuid.uuid4()
        self.trader_id = trader_id
        self.limit_price = price
        self.quantity = quantity
        self.trading_day = trading_day
        self.timestamp = datetime.now()
        self.direct_execution = False
        self.transactions = List[Transaction] = [] 

    def __repr__(self):
        return (f"LimitOrder(order_id={self.order_id}, trader_id={self.trader_id}, "
                f"order_type={self.order_type}, price={self.price}, "
                f"quantity={self.quantity}, timestamp={self.timestamp})")
            
    def get_order_type(self):
        if self.quantity > 0:
            return "buy"
        else:
            return "sell"
        
    def get_quantity_unfilled(self):
        sum_of_transactions = sum([transaction.volume for transaction in self.transactions])
        if self.get_order_type() == "buy":
            return self.quantity - sum_of_transactions 
        else:
            return self.quantity + sum_of_transactions