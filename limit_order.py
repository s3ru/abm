class LimitOrder:
    def __init__(self, order_id, trader_id, order_type, price, quantity, timestamp):
        """
        Initialize a limit order.

        :param order_id: Unique identifier for the order.
        :param trader_id: ID of the trader placing the order.
        :param order_type: Type of the order ('buy' or 'sell').
        :param price: Price at which the order is placed.
        :param quantity: Quantity of the asset to buy or sell.
        :param timestamp: Time when the order was placed.
        """
        self.order_id = order_id
        self.trader_id = trader_id
        self.order_type = order_type  # 'buy' or 'sell'
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

    def __repr__(self):
        return (f"LimitOrder(order_id={self.order_id}, trader_id={self.trader_id}, "
                f"order_type={self.order_type}, price={self.price}, "
                f"quantity={self.quantity}, timestamp={self.timestamp})")