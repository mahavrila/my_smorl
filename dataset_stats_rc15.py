import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import to_pickled_df
import getpass


if __name__ == '__main__':
    data_path = '{}/../div4rec/rc15_data/'.format(os.getcwd())
    # read into pandas dataframe (pandas tabular-like object)
    click_df = pd.read_csv(os.path.join(data_path, 'yoochoose-clicks.dat'), header=None)
    # add names to columns
    click_df.columns = ['session_id', 'timestamp', 'item_id','category']
    # create a column 'valid_session' and assign True/False based on sequence length
    click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['item_id'].size() > 2)
    # remove sessions with False in 'valid_session'
    click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)

    buy_df = pd.read_csv(os.path.join(data_path, 'yoochoose-buys.dat'), header=None)
    buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

    # Stamenkovic sampled a subset of 200000, apparently randomly
    # here we can focus on longer sequences or filter differently,
    # but for reproducibility, we should use his dataset :-/
    sampled_session_id = np.random.choice(click_df.session_id.unique(), 200000, replace=False)
    sampled_click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]

    # is this just for making item_id smaller number??
    item_encoder = LabelEncoder()
    sampled_click_df['item_id'] = item_encoder.fit_transform(sampled_click_df.item_id)

    sampled_buy_df = buy_df.loc[buy_df.session_id.isin(sampled_click_df.session_id)]
    sampled_buy_df['item_id'] = item_encoder.fit_transform(sampled_buy_df.item_id)

    # to_pickled_df(data_path, sampled_clicks=sampled_click_df)
    # to_pickled_df(data_path, sampled_buys=sampled_buy_df)

