from datetime import datetime
import os
import mesa
from file_logger import setup_logger
from helper import get_agent_cols, get_market_cols
from limit_order_market import LimitOrderMarket
import pandas as pd
import numpy as np
from viz import calc_mse, create_viz, get_path, get_price_correlation, save_df_to_excel


# vizualization per run_id and iteration
def is_marginal_trader(data_row):
    return data_row['is_marginal_trader'] == True

def is_non_marginal_trader(data_row):
    return data_row['is_marginal_trader'] == False

def filter_last_day_data(data):
    last_trading_day = data['trading_day'].max()
    return data[data['trading_day'] == last_trading_day]


runtime = current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logger('batch_logger', os.path.join(get_path(runtime)))

# error
# n_days = 50
# n_agents = 50
# range_share_mt = np.linspace(0.05, 0.15, 2)
# print(f"# values for share of marginal traders: {len(range_share_mt)}")
# range_cost_info = range(200, 1001, 200)
# print(f"# values for cost of information: {len(range_cost_info)}")


# quick
n_days = 200
n_agents = 1000
range_share_mt = np.linspace(0.00, 0.3, 6)
print(f"# values for share of marginal traders: {len(range_share_mt)}")
range_cost_info = range(100, 601, 250)
print(f"# values for cost of information: {len(range_cost_info)}")

# prod
# n_days = 200
# n_agents = 500
# range_share_mt = np.linspace(0.05, 0.5, 10)
# print(f"# values for share of marginal traders: {len(range_share_mt)}")
# range_cost_info = range(250, 2501, 250)
# print(f"# values for cost of information: {len(range_cost_info)}")



starting_phase = 10
params = {
    "num_agents": n_agents, #500, # range(250, 1001, 250),
    "n_days": n_days, #200, # 50, # 200,
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

# https://numpy.org/devdocs/reference/generated/numpy.zeros.html
# (rows, cols)
sensitivity_m_corr = np.zeros((len(range_cost_info), len(range_share_mt)))
sensitivity_m_mse = np.zeros((len(range_cost_info), len(range_share_mt)))
sensitivity_m_pnl_mt = np.zeros((len(range_cost_info), len(range_share_mt)))
sensitivity_m_pnl_nmt = np.zeros((len(range_cost_info), len(range_share_mt)))
sensitivity_m_pnl_vola = np.zeros((len(range_cost_info), len(range_share_mt)))

for i, ic in enumerate(range_cost_info):
    # rows 
    for j, mt in enumerate(range_share_mt):
        # cols

        df_filtered =  results_df[results_df['cost_of_information'] == ic]
        df_filtered =  df_filtered[df_filtered['share_of_marginal_traders'] == mt]
        df_filtered = df_filtered.iloc[starting_phase:].reset_index(drop=True)

        df_filterd_m = df_filtered[get_market_cols()].drop_duplicates()
        df_filterd_m.reset_index(drop=True, inplace=True)

        correlation = get_price_correlation(df_filterd_m)
        sensitivity_m_corr[i, j] = correlation
        mse = calc_mse(df_filterd_m)
        sensitivity_m_mse[i, j] = mse

        volatility = df_filterd_m['market_price'].std()
        sensitivity_m_pnl_vola[i, j] = volatility

        df_filtered_a = df_filterd[get_agent_cols()].drop_duplicates()
        df_filtered_a.reset_index(drop=True, inplace=True)
        df_filterd = filter_last_day_data(df_filtered_a)
        avg_pnl_marginal = df_filterd[is_marginal_trader(df_filtered_a)]['PnL'].mean()
        sensitivity_m_pnl_mt[i, j] = round(avg_pnl_marginal, 2)

        avg_pnl_non_marginal = df_filterd[is_non_marginal_trader(df_filtered_a)]['PnL'].mean()
        sensitivity_m_pnl_nmt[i, j] = round(avg_pnl_non_marginal, 2)
        
sensitivity_df_corr = pd.DataFrame(
    sensitivity_m_corr,
    index=[f"ic={v:.0f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)
save_df_to_excel(sensitivity_df_corr, runtime, "corr")

sensitivity_df_mse = pd.DataFrame(
    sensitivity_m_mse,
    index=[f"ic={v:.0f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)
save_df_to_excel(sensitivity_df_mse, runtime, "mse")

sensitivity_df_pnl_mt = pd.DataFrame(
    sensitivity_m_pnl_mt,
    index=[f"ic={v:.0f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)
save_df_to_excel(sensitivity_df_pnl_mt, runtime, "pnl")

sensitivity_df_pnl_vola = pd.DataFrame(
    sensitivity_m_pnl_vola,
    index=[f"ic={v:.0f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)
save_df_to_excel(sensitivity_df_pnl_vola, runtime, "pnl")

sensitivity_df_pnl_nmt = pd.DataFrame(
    sensitivity_m_pnl_nmt,
    index=[f"ic={v:.0f}" for v in range_cost_info],
    columns=[f"mt={v:.0%}" for v in range_share_mt]
)
save_df_to_excel(sensitivity_df_pnl_nmt, runtime, "pnl")





for (run_id, iteration), group_data in results_df.groupby(["RunId", "iteration"]):

    # sel_columns = ['RunId', 'iteration', 'Step', 'num_agents', 'n_days', 'trading_day', 'market_price', 'info_event',
    #            'true_value', 'bid_ask_spread', 'volume', 'shares_outstanding', 'share_of_marginal_traders',
    #            'rel_distance_sell_orders', 'rel_distance_buy_orders', 'cost_of_information', 'is_marginal_trader', 'PnL', 'skill', 'num_bought_information', 'info_budget_constraint']
    # print(f"Generating visualization for RunId: {run_id}, Iteration: {iteration}")

    rows_for_eval = group_data # group_data[sel_columns].drop_duplicates()
    rows_for_eval = rows_for_eval.iloc[starting_phase:].reset_index(drop=True)

    create_viz(rows_for_eval, runtime)


    df_market = rows_for_eval[get_market_cols()].drop_duplicates()
    df_market.reset_index(drop=True, inplace=True)
    avg_rel_distance_buy_orders = df_market['rel_distance_buy_orders'].mean()
    avg_rel_distance_sell_orders = df_market['rel_distance_sell_orders'].mean()
    # print(f"Avg Rel Distance Buy Orders: {avg_rel_distance_buy_orders:.2f}, Avg Rel Distance Sell Orders: {avg_rel_distance_sell_orders:.2f}")
    
    # median distance buy orders
    median_rel_distance_buy_orders = df_market['rel_distance_buy_orders'].median()
    median_rel_distance_sell_orders = df_market['rel_distance_sell_orders'].median()
    # print(f"Median Rel Distance Buy Orders: {median_rel_distance_buy_orders:.2f}, Median Rel Distance Sell Orders: {median_rel_distance_sell_orders:.2f}")


    num_info_events = df_market['info_event'].sum()
    print(f"Number of Information Events: {num_info_events}")
    volatility_mp = df_market['market_price'].std()
    print(f"Volatility of Market Prices: {volatility_mp:.2f}")
    volatility_tv = df_market['true_value'].std()
    print(f"Volatility of True Value: {volatility_tv:.2f}")


    # Filter for the last trading day
    df_agents = rows_for_eval[get_agent_cols()].drop_duplicates()
    df_agents.reset_index(drop=True, inplace=True)
    last_day_data = filter_last_day_data(df_agents)

    # Calculate average PnL for marginal and non-marginal traders
    avg_pnl_marginal = last_day_data[is_marginal_trader(last_day_data)]['PnL'].mean()
    avg_pnl_non_marginal = last_day_data[is_non_marginal_trader(last_day_data)]['PnL'].mean()


    avg_info_budget_constraint = df_agents[is_marginal_trader(df_agents)]['info_budget_constraint'].mean()
    print(f"Average of budget_constraints marginal traders': {avg_info_budget_constraint:.2%}")

    print(f"RunId: {run_id}, Iteration: {iteration}")
    print(f"Average PnL of Marginal Traders on Last Trading Day: {avg_pnl_marginal:.2f}")
    print(f"Average PnL of Non-Marginal Traders on Last Trading Day: {avg_pnl_non_marginal:.2f}")


