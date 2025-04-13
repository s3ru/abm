

from limit_order_market import LimitOrderMarket


model = LimitOrderMarket(num_agents=1000, n_days=200, decision_threshold=0.05, order_expiration=10)
model.run_model()


print(model.transactions)