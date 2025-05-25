from enum import Enum
from typing import List
import uuid
from datetime import datetime


class OrderStatus(Enum):
    FILLED = "Filled"
    PARTIALLY_FILLED = "Partially Filled"
    CANCELED = "Canceled"
    OPEN = "Open"


class LimitOrder:
    def __init__(self, trader_id: int, price: float, quantity: int, trading_day: int, is_trader_mt: bool = False):
        """
        Initialize a limit order.
        """
        self.order_id = uuid.uuid4()
        self.trader_id = trader_id
        self.is_trader_mt = is_trader_mt
        self.limit_price = price
        self.quantity = quantity
        self.trading_day = trading_day
        self.timestamp = datetime.now()
        self.transactions = [] 
        self.is_canceled = False

    def __repr__(self):
        return (f"LimitOrder(order_id={self.order_id}, trader_id={self.trader_id}, "
                f"order_type={self.get_order_type()}, price={self.limit_price}, "
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
        
    def get_avg_execution_price(self):
        if len(self.transactions) == 0:
            return 0
        return sum([transaction.price * transaction.volume for transaction in self.transactions]) / sum([transaction.volume for transaction in self.transactions])
        
    def get_status(self):
        if self.is_canceled:
            return OrderStatus.CANCELED
        elif self.get_quantity_unfilled() == 0:
            return OrderStatus.FILLED
        elif abs(self.get_quantity_unfilled()) > 0:
            return OrderStatus.PARTIALLY_FILLED
        else:
            return OrderStatus.OPEN
        
    
class Transaction:
    def __init__(self, price: float, volume: float, buyer_order: LimitOrder, seller_order: LimitOrder, trading_day: int):
        """
        Represents a transaction in the market.
        """
        self.price = price
        self.volume = volume
        self.buyer_order = buyer_order
        self.seller_order = seller_order
        self.trading_day = trading_day
        self.mt_involvement = buyer_order.is_trader_mt or seller_order.is_trader_mt

    def get_initiator(self):
        if self.buyer_order.timestamp > self.seller_order.timestamp:
            return "buy"
        else:
            return "sell"

    def __repr__(self) -> str:
        return (f"Transaction(price={self.price}, volume={self.volume}, "
                f"buyer_id={self.buyer_order.trader_id}, seller_id={self.seller_order.trader_id}, "
                f"trading_day={self.trading_day})")