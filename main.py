

from limit_order_market import LimitOrderMarket
from viz import create_viz


starting_phase = 10

model = LimitOrderMarket(
    num_agents=500,
    n_days=20,
    start_true_value=100,
    decision_threshold=0.05, 
    order_expiration=10,
    noise_range=0.05,
    wealth_min=5000,
    event_info_frequency = 0.2,
    event_info_intensity = 1,
    event_liquidity_frequency = 0.05,
    event_liquidity_intensity = 10,
    starting_phase=starting_phase,
)
model.run_model()
# create_viz(model)

# print(model.transactions)





