

from limit_order_market import LimitOrderMarket


model = LimitOrderMarket(
    num_agents=500,
    n_days=200,
    decision_threshold=0.05, 
    order_expiration=10,
    noise_range=0.05,
    wealth_min=5000
)
model.run_model()


print(model.transactions)