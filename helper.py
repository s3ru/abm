import numpy as np


def lognormal(mu, sigma):
    return np.random.lognormal(mean=mu, sigma=sigma)

def normal(mu, sigma):
    return np.random.normal(loc=mu, scale=sigma)

def poisson(lmbda):
    return np.random.poisson(lam=lmbda)

def getRandomUniform(low, high):
    return np.random.uniform(low, high)

def get_agent_cols():
    agent_cols = ['RunId', 'iteration', 'trading_day', 'AgentID', 'is_marginal_trader',
                  'PnL', 'wealth', 'cash', 'cash_chg_liquidity_events', 'skill', 'num_of_shares', 'informed',
                  'num_bought_information', 'info_budget_constraint', 'stocks_sold', 'stocks_sold_avg_price', 'stocks_bought', 'stocks_bought_avg_price']
    return agent_cols

def get_market_cols():
    market_cols = ['RunId', 'iteration', 'trading_day', 'trading_date', 'vpin', 'lob_imbalance', 'market_price', 'true_value', 'bid_ask_spread', 'open_price', 'high_price', 'low_price', 'close_price',
                   'info_event', 'volume', 'shares_outstanding', 'share_of_marginal_traders',  'buy_orders', 'sell_orders',
                   'rel_distance_sell_orders', 'rel_distance_buy_orders', 'cost_of_information', 'volume_mt_involment']
    
    return market_cols