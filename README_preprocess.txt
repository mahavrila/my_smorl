Preprocessing of rc15 dataset in original paper goes like this:  

  |<---- youchoose_buys.dat  
  |<---- youchoose_clicks.dat  
  |  
  V  
sample_data_rc15.py  
  |  
  |----> sampled_clicks.df  
  |----> sampled_byus.df  
  V
merge_and_sort_rc15.py
  |
  |----> sampled_sessions.csv
  |----> sampled_sessions.df
  |
  |---------------------------------> nov_items_rewards.py
  |                                     |
  V                                     |----> binary_nov_reward.csv
split_data.py                           |----> less-popupar_items.plk
  |
  |----> sampled_test.df
  |----> sampled_val.df
  |----> sampled_train.df
  |
  V
replay_buffer_rc15.py
  |
  |
  |----> replay_buffer_test.df
  |----> replay_buffer_val.df
  |----> replay_buffer.df

Preprocessing starts with original datasets from recsys-challenge-2015:
https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015/
Datasets for byus and clicks are:
 - youchoose_buys.dat
 - youchoose_clicks.dat
 
1. step: sample_data_rc15.py adds column descriptions; remove short sessions  
         with length < 3; take random 200000 sessions from clicks dataset and
         saves data as:
 - sampled_clicks.df
 - sampled_buys.dfreplay_buffer_val.df

2. step: merge_and_sort_rc15.py drops unnecessary columns; add column identifying  
         whether it's buy or click and merge buys and clicks to single dataset; 
         sort dataset based on session_id and timestamp - i.e. chronologically
 - sampled_sessions.df
 - sampled_sessions.csv

3. step: split_data.py just splits data to train, test and val datasets
 - sampled_train.df 
 - sampled_test.df
 - sampled_val.df
 
4. step: replay_buffer_rc15.py goes through chronological sessions data and
         reformat it to specific format containing state, action; next_state
         and some more columns.
 - replay_buffer.df
 - replay_buffer_test.df
 - replay_buffer_val.df
 
5. step: nov_items_rewards.py takes sampled sessions.df and create two more
         files necessary for statistical evaluation of models.
 - binary_nov_reward.csv
 - less-popupar_items.plk
         
                           