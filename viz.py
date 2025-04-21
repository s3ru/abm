
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
import seaborn as sns
print(sns.__version__)
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

# Get the directory of the current file
current_folder = os.path.dirname(os.path.abspath(__file__))

def create_viz(df):
    # print("Creating visualization...")

    # starting_phase = df.iloc[0]["starting_phase"]
    # plot market_price on searborn line chart
    model_dfs = df
    # model_dfs = model_dfs.iloc[starting_phase:] # remove first x days to avoid noise
    # model_dfs = model_dfs.reset_index(drop=True)

    # append information events to the dataframe
    df_info = model_dfs.info_event # [starting_phase:]
    df_info = df_info[1:] + [False] # shift for visualization
    df_info.reset_index(drop=True, inplace=True)
    model_dfs['info_event'] = df_info
    model_dfs['info_event'] = model_dfs['info_event'].fillna(False)
    # model_dfs['info_event'] = df_info

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path_csv = os.path.join(current_folder, 'data', f"df__{current_time}.xlsx")
    model_dfs.to_excel(file_path_csv, index=True)

    # print head
    # print(model_dfs.head())

    # line plot with column 'market_price' and 'true_value' 


    agents = df.iloc[0]["num_agents"]
    share_mt = df.iloc[0]["share_of_marginal_traders"]
    ic = df.iloc[0]["cost_of_information"]
    # eff = evaluate_market_efficiency(df)
    mse = calc_mse(df)
    corr = get_price_correlation(df)

    selected_columns = ['trading_day', 'market_price', 'true_value', 'volume']
    filtered_df = model_dfs[selected_columns]

    # Create the figure and axes
    fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # fig.suptitle('Market Price and Volume Over Time')

    # Line chart for market_price
    sns.lineplot(data=filtered_df, x='trading_day', y='market_price', label='Market Price', ax=axs[0], color='steelblue', alpha=0.7)
    sns.lineplot(data=filtered_df, x='trading_day', y='true_value', label='True Value', ax=axs[0], color='seagreen', alpha=0.7)
    axs[0].set_title(f"Prices Over Time [Agents: {agents}, MT: {share_mt:.0%}, IC: {ic}, Corr: {corr:.2f}, MSE: {mse:.1f} ]")
    axs[0].set_ylabel('Price')


    # Add dots on the true_value line for information events
    event_days = filtered_df[model_dfs['info_event']]['trading_day']
    event_values = filtered_df[model_dfs['info_event']]['true_value']
    sns.scatterplot(x=event_days, y=event_values, ax=axs[0], color='darkgreen', label='Information Event', s=50)


    # Update legend
    h, l = axs[0].get_legend_handles_labels()
    labels = ["market price", "true value", "information event"]
    axs[0].legend(h, l, title="Legend", loc="upper left")



    # Histogram for volume
    n_days_value = df.iloc[0]["n_days"]
    sns.histplot(data=filtered_df, x='trading_day', weights='volume', bins=round(n_days_value/2), ax=axs[1], kde=False, color='grey', alpha=0.5)
    axs[1].set_title('Volume Over Time')
    axs[1].set_ylabel('Volume')
    axs[1].set_xlabel('Trading Day')

    # Save the plot to a file
    file_path = os.path.join(current_folder, 'img', f"price_chart__{current_time}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

    file_path_xlsx = os.path.join(current_folder, 'data', f"df__{current_time}.xlsx")
    df.to_excel(file_path_xlsx, index=True)
    plt.show()


def get_price_correlation(df):
    prices = df["market_price"].values
    true_vals = df["true_value"].values

    # calculate pearson correlation 
    correlation = np.corrcoef(prices, true_vals)[0, 1]
    # print(f"Pearson-Korrelation Marktpreis vs. fundamentaler Wert: {correlation:.2f}")

    # avg_deviation = np.mean(np.abs(prices - true_vals))
    # avg_true_value = np.mean(true_vals)
    # rel_efficiency = 1 - (avg_deviation / avg_true_value)
    # print(f"Durchschnittliche Abweichung Marktpreis vs. fundamentaler Wert: {rel_efficiency:.2f}")
    return correlation

def evaluate_market_efficiency(df):
    prices = df["market_price"].values
    true_vals = df["true_value"].values

    
    avg_deviation = np.mean(np.abs(prices - true_vals))
    avg_true_value = np.mean(true_vals)
    rel_efficiency = 1 - (avg_deviation / avg_true_value)
    # print(f"Durchschnittliche Abweichung Marktpreis vs. fundamentaler Wert: {rel_efficiency:.2f}")
    return rel_efficiency

def calc_mse(df): 
    prices = df["market_price"].values
    true_vals = df["true_value"].values
    mse = mean_squared_error(prices, true_vals)
    return mse

def save_df_to_excel(df, prefix):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(current_folder, 'data', f"{prefix}__{current_time}.xlsx")
    df.to_excel(file_path, index=True)
    print(f"DataFrame saved to {file_path}")