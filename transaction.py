from limit_order import LimitOrder


class Transaction:
    def __init__(self, price: float, volume: float, buyer_order: LimitOrder, seller_order: LimitOrder, trading_day: int):
        """
        Represents a transaction in the market.

        :param price: The transaction price.
        :param volume: The volume of the transaction.
        :param buyer_id: The ID of the buyer.
        :param seller_id: The ID of the seller.
        :param trading_day: The trading_day of the transaction.
        """
        self.price = price
        self.volume = volume
        self.buyer_order = buyer_order
        self.seller_order = seller_order
        self.trading_day = trading_day

    def __repr__(self) -> str:
        return (f"Transaction(price={self.price}, volume={self.volume}, "
                f"buyer_id={self.buyer_order.trader_id}, seller_id={self.seller_order.trader_id}, "
                f"trading_day={self.trading_day})")