
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
import seaborn as sns

from helper import get_agent_cols, get_market_cols
print(sns.__version__)
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import mplfinance as mpf

# col names model parameters
run_id_col = 'RunId'
share_of_marginal_traders_col = 'share_of_marginal_traders'
cost_of_information_col = 'cost_of_information'
num_agents_col = 'num_agents'
n_days_col = 'n_days'


# Get the directory of the current file
current_folder = os.path.dirname(os.path.abspath(__file__))

def create_viz(df, runtime):
    create_price_chart(df, runtime)
    create_skill_histogram(df, runtime)
    create_pnl_boxplot(df, runtime)
    create_candlestick_chart(df, runtime)


def create_candlestick_chart(df, runtime):
    share_mt = df.iloc[0][share_of_marginal_traders_col]
    ic = df.iloc[0][cost_of_information_col]
    df = df[['trading_date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
    df = df.drop_duplicates()

    df = df.rename(columns={
        'trading_date': 'Date',
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume'
    })

    # Ensure the Date column is a datetime object and set it as the index
    # df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    file_path = os.path.join(get_path(runtime), f"candlestick_chart.png")
    # Plot the candlestick chart
    mpf.plot(
        df,
        type='candle',
        volume=True,
        title=f"Candlestick Chart [ MT: {share_mt:.0%}, IC: {ic} ]",
        style='yahoo',
        savefig=dict(fname=file_path, dpi=300, bbox_inches='tight')
    )


def create_pnl_boxplot(df, runtime):

    share_mt = df.iloc[0][share_of_marginal_traders_col]
    ic = df.iloc[0][cost_of_information_col]

    df = df[get_agent_cols()]
    df = df.drop_duplicates()

    # Filter for the last trading day
    last_trading_day = df['trading_day'].max()
    df = df[df['trading_day'] == last_trading_day]

    df['Trader Type'] = df['is_marginal_trader'].apply(lambda x: 'Marginal' if x else 'Non-Marginal')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='Trader Type',
        y='PnL',
        order=['Non-Marginal', 'Marginal'],  # Ensure Non-Marginal is on the left, Marginal on the right
        palette={'Non-Marginal': 'darkgrey', 'Marginal': 'red'}  # Set specific colors
    )

    # Add labels and title
    plt.title(f'PnL Distribution for Marginal and Non-Marginal Traders [ MT: {share_mt:.0%}, IC: {ic} ]')
    plt.xlabel('Trader Type')
    plt.ylabel('PnL')
    plt.grid(axis='y', alpha=0.3)
    save_plt_as_img(plt, runtime, f"pnl_boxplot_{df.iloc[0]['RunId']}{df.iloc[0]['iteration']}")



def create_skill_histogram(df, runtime):
        # Filter for the last trading day

    share_mt = df.iloc[0]["share_of_marginal_traders"]

    df = df[get_agent_cols()]
    df = df.drop_duplicates()
    
    run_id = df.iloc[0]["RunId"]
    iteration = df.iloc[0]["iteration"]
    last_trading_day = df['trading_day'].max()
    last_day_data = df[df['trading_day'] == last_trading_day]

    # Filter data for marginal and non-marginal traders
    marginal_traders = last_day_data[last_day_data['is_marginal_trader'] == True]
    non_marginal_traders = last_day_data[last_day_data['is_marginal_trader'] == False]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(non_marginal_traders['skill'], color='darkgrey', label='Non-Marginal Traders', kde=False, bins=20, alpha=0.7)
    sns.histplot(marginal_traders['skill'], color='red', label='Marginal Traders', kde=False, bins=20, alpha=1)


    # Add labels and legend
    plt.title(f'Skill Distribution by Trader Type [ MT: {share_mt:.0%} ]')
    plt.xlabel('Skill')
    plt.ylabel('Count')
    plt.legend(title='Trader Type')

    save_plt_as_img(plt, runtime, f"skill_histogram_{run_id}{iteration}")
    save_df_to_excel(df, runtime, f"agent_data_{run_id}{iteration}")


    # plt.show()

def create_price_chart(df, runtime):
    # print("Creating visualization...")

    # starting_phase = df.iloc[0]["starting_phase"]
    # plot market_price on searborn line chart
    df = df
    # df = df.iloc[starting_phase:] # remove first x days to avoid noise
    # df = df.reset_index(drop=True)


    # model parameters 
    run_id = df.iloc[0][run_id_col]
    iteration = df.iloc[0]["iteration"]
    agents = df.iloc[0][num_agents_col]
    share_mt = df.iloc[0][share_of_marginal_traders_col]
    ic = df.iloc[0][cost_of_information_col]
    n_days_value = df.iloc[0][n_days_col]

    

    info_event_col = 'info_event'
    trading_day_col = 'trading_day'
    market_price_col = 'market_price'
    true_value_col = 'true_value'
    volume_col = 'volume'

    df = df[get_market_cols()].drop_duplicates()
    df.reset_index(drop=True, inplace=True)

    # append information events to the dataframe
    df_info = df.info_event # [starting_phase:]
    df_info = df_info[1:] + [False] # shift for visualization
    df_info.reset_index(drop=True, inplace=True)
    df[info_event_col] = df_info
    df[info_event_col] = df[info_event_col].fillna(False)
    # df['info_event'] = df_info

 
    # eff = evaluate_market_efficiency(df)
    mse = calc_mse(df)
    corr = get_price_correlation(df)

    selected_columns = ['trading_day', 'market_price', 'true_value', 'volume']
    filtered_df = df[selected_columns]

    # Create the figure and axes
    fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # fig.suptitle('Market Price and Volume Over Time')

    # Line chart for market_price
    sns.lineplot(data=filtered_df, x='trading_day', y='market_price', label='Market Price', ax=axs[0], color='steelblue', alpha=0.7)
    sns.lineplot(data=filtered_df, x='trading_day', y='true_value', label='True Value', ax=axs[0], color='seagreen', alpha=0.7)
    axs[0].set_title(f"Prices Over Time [Agents: {agents}, MT: {share_mt:.0%}, IC: {ic}, Corr: {corr:.2f}, MSE: {mse:.1f} ]")
    axs[0].set_ylabel('Price')


    # Add dots on the true_value line for information events
    event_days = filtered_df[df['info_event']]['trading_day']
    event_values = filtered_df[df['info_event']]['true_value']
    sns.scatterplot(x=event_days, y=event_values, ax=axs[0], color='darkgreen', label='Information Event', s=50)


    # Update legend
    h, l = axs[0].get_legend_handles_labels()
    labels = ["market price", "true value", "information event"]
    axs[0].legend(h, l, title="Legend", loc="upper left")


    # Histogram for volume
    
    sns.histplot(data=filtered_df, x='trading_day', weights='volume', bins=round(n_days_value/2), ax=axs[1], kde=False, color='grey', alpha=0.5)
    axs[1].set_title('Volume Over Time')
    axs[1].set_ylabel('Volume')
    axs[1].set_xlabel('Trading Day')


    # save results to filesystem
    save_plt_as_img(plt, runtime, f"price_chart_{run_id}{iteration}")
    save_df_to_excel(df, runtime, f"price_data_{run_id}{iteration}")
    # plt.show()


def get_price_correlation_with_lag(df, lag):
    prices = df["market_price"].values
    true_vals = df["true_value"].values
    prices = prices.shift(lag)
    return true_vals.corr(prices)

def get_price_correlation(df):
    prices = df["market_price"].values
    true_vals = df["true_value"].values
    return np.corrcoef(prices, true_vals)[0, 1]

# def evaluate_market_efficiency(df):
#     prices = df["market_price"].values
#     true_vals = df["true_value"].values
    
#     avg_deviation = np.mean(np.abs(prices - true_vals))
#     avg_true_value = np.mean(true_vals)
#     rel_efficiency = 1 - (avg_deviation / avg_true_value)
#     # print(f"Durchschnittliche Abweichung Marktpreis vs. fundamentaler Wert: {rel_efficiency:.2f}")
#     return rel_efficiency

def calc_mse(df): 
    prices = df["market_price"].values
    true_vals = df["true_value"].values
    mse = mean_squared_error(prices, true_vals)
    return mse

def save_df_to_excel(df, runtime, prefix):
    file_path = os.path.join(get_path(runtime), f"{prefix}.xlsx")
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html
    df.to_excel(file_path, index=True)
    

def save_plt_as_img(plt, runtime, prefix):
    file_path = os.path.join(get_path(runtime), f"{prefix}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')


def get_path(runtime):
    path = os.path.join(current_folder, "data", runtime)
    os.makedirs(path, exist_ok=True)
    return path