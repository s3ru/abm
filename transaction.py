from datetime import datetime

class Transaction:
    def __init__(self, price: float, volume: float, buyer_id: int, seller_id: int, timestamp: datetime):
        """
        Represents a transaction in the market.

        :param price: The transaction price.
        :param volume: The volume of the transaction.
        :param buyer_id: The ID of the buyer.
        :param seller_id: The ID of the seller.
        :param timestamp: The timestamp of the transaction.
        """
        self.price = price
        self.volume = volume
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (f"Transaction(price={self.price}, volume={self.volume}, "
                f"buyer_id={self.buyer_id}, seller_id={self.seller_id}, "
                f"timestamp={self.timestamp})")