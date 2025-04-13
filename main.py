

from limit_order_market import LimitOrderMarket


model = LimitOrderMarket(num_agents=50, n_days=100, decision_threshold=0.05, order_expiration=10)
model.run_model()


print(model.transactions)