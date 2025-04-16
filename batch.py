import mesa
from limit_order_market import LimitOrderMarket
import pandas as pd

from viz import create_viz


params = {
    "num_agents": 250, #  range(250, 1001, 250),
    "n_days": 100 # 50, # 200,
    # "start_true_value": 100,
    # "decision_threshold": 0.05,
    # "order_expiration": 10,
    # "noise_range": 0.05,
    # "wealth_min": 5000,
    # "event_info_frequency": 0.2,
    # "event_info_intensity": 1,
    # "event_liquidity_frequency": 0.05,
    # "event_liquidity_intensity": 10,
    # "starting_phase": 10
}

results = mesa.batch_run(
    LimitOrderMarket,
    parameters=params,
    iterations=1,
    max_steps= params["n_days"],
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)

results_df = pd.DataFrame(results)
# print(f"The results have {len(results)} rows.")
# print(f"The columns of the data frame are {list(results_df.keys())}.")
# print(results_df.head())

# Iterate over each unique combination of RunId and iteration
for (run_id, iteration), group_data in results_df.groupby(["RunId", "iteration"]):

    sel_columns = ['Step', 'num_agents', 'n_days', 'trading_day', 'market_price', 'info_event', 'true_value', 'bid_ask_spread', 'volume']
    # print(f"Generating visualization for RunId: {run_id}, Iteration: {iteration}")

    rows_for_eval = group_data[sel_columns].drop_duplicates()
    rows_for_eval = rows_for_eval.iloc[10:].reset_index(drop=True)
    
    # Create the visualization for the current group
    create_viz(rows_for_eval)


run_ids = results_df["RunId"].unique()
for run_id in run_ids:
    print(f"Generating visualization for RunId: {run_id}")
    
    # Filter the DataFrame for the current RunId
    run_data = results_df[results_df["RunId"] == run_id]
    
    # do calculations for the current run_id