

from limit_order_market import LimitOrderMarket


starting_phase = 10

model = LimitOrderMarket(
    num_agents=500,
    n_days=50,
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


# print(model.transactions)

# plot market_price on searborn line chart
model_dfs = model.data_collector.get_model_vars_dataframe()
model_dfs = model_dfs.iloc[starting_phase:] # remove first 5 days to avoid noise
model_dfs = model_dfs.reset_index(drop=True)

# append information events to the dataframe
df_info = model.information_events[starting_phase:]
df_info = df_info[1:] + [False] # shift for visualization
model_dfs['information_events'] = df_info

# print head
print(model_dfs.head())





# line plot with column 'market_price' and 'true_value' 
import seaborn as sns
print(sns.__version__)
import matplotlib.pyplot as plt
import pandas as pd

selected_columns = ['trading_day', 'market_price', 'true_value', 'volume']
filtered_df = model_dfs[selected_columns]

# Create the figure and axes
fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
# fig.suptitle('Market Price and Volume Over Time')

# Line chart for market_price
sns.lineplot(data=filtered_df, x='trading_day', y='market_price', label='Market Price', ax=axs[0], color='steelblue', alpha=0.7)
sns.lineplot(data=filtered_df, x='trading_day', y='true_value', label='True Value', ax=axs[0], color='seagreen', alpha=0.7)
axs[0].set_title('Market Price and True Value Over Time')
axs[0].set_ylabel('Price')


# Add dots on the true_value line for information events
event_days = filtered_df[model_dfs['information_events']]['trading_day']
event_values = filtered_df[model_dfs['information_events']]['true_value']
sns.scatterplot(x=event_days, y=event_values, ax=axs[0], color='darkgreen', label='Information Event', s=50)


# Update legend
h, l = axs[0].get_legend_handles_labels()
labels = ["market price", "true value", "information event"]
axs[0].legend(h, l, title="Legend", loc="upper left")



# Histogram for volume
sns.histplot(data=filtered_df, x='trading_day', weights='volume', bins=round(model.n_days/2), ax=axs[1], kde=False, color='grey', alpha=0.5)
axs[1].set_title('Volume Over Time')
axs[1].set_ylabel('Volume')
axs[1].set_xlabel('Trading Day')


plt.show()





