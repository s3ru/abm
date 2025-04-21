import mesa
from limit_order_market import LimitOrderMarket
import pandas as pd
import numpy as np
from viz import calc_mse, create_viz, get_price_correlation, save_df_to_excel


# quick
range_share_mt = np.linspace(0.05, 0.15, 2)
print(len(range_share_mt))
range_cost_info = range(500, 1001, 500)
print(len(range_cost_info))

# prod
# range_share_mt = np.linspace(0.05, 0.5, 10)
# print(len(range_share_mt))
# range_cost_info = range(500, 5001, 500)
# print(len(range_cost_info))

starting_phase = 10
params = {
    "num_agents": 500, #500, # range(250, 1001, 250),
    "n_days": 150, #200, # 50, # 200,
    "share_of_marginal_traders": range_share_mt,
    "cost_of_information": range_cost_info,
    # "start_true_value": 100,
    # "decision_threshold": 0.05,
    # "order_expiration": 10,
    # "noise_range": 0.05,
    # "wealth_min": 5000,
    "event_info_frequency": 0.1,
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
sensitivity_m_corr = np.zeros((len(range_share_mt), len(range_cost_info)))
sensitivity_m_mse = np.zeros((len(range_share_mt), len(range_cost_info)))


for i, ic in enumerate(range_cost_info):
    for j, mt in enumerate(range_share_mt):
        # Example calculation: Replace this with your model's calculation
        df_filtered =  results_df[results_df['cost_of_information'] == ic]
        df_filtered =  df_filtered[df_filtered['share_of_marginal_traders'] == mt]

        correlation = get_price_correlation(df_filtered)
        sensitivity_m_corr[i, j] = correlation
        mse = calc_mse(df_filtered)
        sensitivity_m_mse[i, j] = mse
        # print(f"Correlation: {correlation:.2f}, MSE: {mse:.2f}")



sensitivity_df_corr = pd.DataFrame(
    sensitivity_m_corr,
    index=[f"ic={v:.2f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)

sensitivity_df_mse = pd.DataFrame(
    sensitivity_m_corr,
    index=[f"ic={v:.2f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)

save_df_to_excel(sensitivity_df_corr, "corr")
save_df_to_excel(sensitivity_df_mse, "mse")

# Iterate over each unique combination of RunId and iteration
for (run_id, iteration), group_data in results_df.groupby(["RunId", "iteration"]):

    sel_columns = ['Step', 'num_agents', 'n_days', 'trading_day', 'market_price', 'info_event',
                'true_value', 'bid_ask_spread', 'volume', 'shares_outstanding', 'share_of_marginal_traders',
                'rel_distance_sell_orders', 'rel_distance_buy_orders', 'cost_of_information', 'is_marginal_trader', 'PnL', 'num_bought_information', 'num_budget_contstraint']
    # print(f"Generating visualization for RunId: {run_id}, Iteration: {iteration}")

    rows_for_eval = group_data[sel_columns].drop_duplicates()
    rows_for_eval = rows_for_eval.iloc[starting_phase:].reset_index(drop=True)

    avg_rel_distance_buy_orders = rows_for_eval['rel_distance_buy_orders'].mean()
    avg_rel_distance_sell_orders = rows_for_eval['rel_distance_sell_orders'].mean()
    # print(f"Avg Rel Distance Buy Orders: {avg_rel_distance_buy_orders:.2f}, Avg Rel Distance Sell Orders: {avg_rel_distance_sell_orders:.2f}")
    
    # median distance buy orders
    median_rel_distance_buy_orders = rows_for_eval['rel_distance_buy_orders'].median()
    median_rel_distance_sell_orders = rows_for_eval['rel_distance_sell_orders'].median()
    # print(f"Median Rel Distance Buy Orders: {median_rel_distance_buy_orders:.2f}, Median Rel Distance Sell Orders: {median_rel_distance_sell_orders:.2f}")


    num_info_events = rows_for_eval['info_event'].sum()
    print(f"Number of Information Events: {num_info_events}")
    volatility_mp = rows_for_eval['market_price'].std()
    print(f"Volatility of Market Prices: {volatility_mp:.2f}")
    volatility_tv = rows_for_eval['true_value'].std()
    print(f"Volatility of True Value: {volatility_tv:.2f}")


    # Filter for the last trading day
    last_trading_day = rows_for_eval['trading_day'].max()
    last_day_data = rows_for_eval[rows_for_eval['trading_day'] == last_trading_day]

    # Calculate average PnL for marginal and non-marginal traders
    avg_pnl_marginal = last_day_data[last_day_data['is_marginal_trader'] == True]['PnL'].mean()
    avg_pnl_non_marginal = last_day_data[last_day_data['is_marginal_trader'] == False]['PnL'].mean()

    print(f"RunId: {run_id}, Iteration: {iteration}")
    print(f"Average PnL of Marginal Traders on Last Trading Day: {avg_pnl_marginal:.2f}")
    print(f"Average PnL of Non-Marginal Traders on Last Trading Day: {avg_pnl_non_marginal:.2f}")


    # Create the visualization for the current group
    create_viz(rows_for_eval)


run_ids = results_df["RunId"].unique()
for run_id in run_ids:
    print(f"Generating visualization for RunId: {run_id}")
    
    # Filter the DataFrame for the current RunId
    run_data = results_df[results_df["RunId"] == run_id]
    
    # do calculations for the current run_id