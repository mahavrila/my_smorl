import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import dok_matrix
import pandas as pd
import sys
import pickle


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def set_device(id):
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device(f"cuda:{id}")
    else:
        device = torch.device("cpu")
    return device


def prepare_dataloader(data_path, batch_size, dataset=None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')
    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.long)).long()
    next_states = replay_buffer_dic['next_state'].values()
    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.long)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.long)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool))
    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def prepare_dataloader_skips(data_path, batch_size, dataset=None, skip=0):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + f'replay_buffer_train_skip={skip}.df')
    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + f'replay_buffer_val_skip={skip}.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + f'replay_buffer_test_skip={skip}.df')
    else:
        raise Exception('dataset has to be either None, val or test')
    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.long)).long()
    next_states = replay_buffer_dic['next_state'].values()
    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.long)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.long)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool))
    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def get_one_hot_item_sess(data_path):
    sorted_events = pd.read_csv(data_path + 'sorted_events.csv')
    item_sess_one_hot = dok_matrix(
        shape=(sorted_events.item_id.max() + 1, sorted_events.session_id.max() + 1),
        dtype=np.int32
    )
    for item_id, session_id in zip(sorted_events.item_id.values, sorted_events.session_id.values):
        item_sess_one_hot[item_id, session_id] = 1
    return item_sess_one_hot


def calculate_hit(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def get_last_clicks(state, len_state, k=3):
    if len_state < k:
        return state[:len_state]
    else:
        return state[len_state - k: len_state]


def calculate_session_repetitions(session_ids, top_20_preds):
    top_10_preds = [x[-10:] for x in top_20_preds]
    top_5_preds = [x[-5:] for x in top_20_preds]
    rpt_df = pd.DataFrame({
        'session_id': session_ids,
        'top_20_preds': top_20_preds,
        'top_10_preds': top_10_preds,
        'top_5_preds': top_5_preds
    })
    num_rpts_20 = 0
    num_rpts_10 = 0
    num_rpts_5 = 0
    rpt_groups = rpt_df.groupby('session_id')
    for _, group in rpt_groups:
        all_top_20 = np.concatenate(group.top_20_preds.values)
        all_top_10 = np.concatenate(group.top_10_preds.values)
        all_top_5 = np.concatenate(group.top_5_preds.values)
        num_rpts_20 += len(all_top_20) - len(np.unique(all_top_20))
        num_rpts_10 += len(all_top_10) - len(np.unique(all_top_10))
        num_rpts_5 += len(all_top_5) - len(np.unique(all_top_5))
    num_rpts_20 /= rpt_groups.ngroups
    num_rpts_10 /= rpt_groups.ngroups
    num_rpts_5 /= rpt_groups.ngroups
    return num_rpts_5, num_rpts_10, num_rpts_20


def eval_cov_nov_cov(all_top_20, item_num, data_path):
    all_top_1 = [x[-1] for x in all_top_20]
    all_top_5 = [x[-5:] for x in all_top_20]
    all_top_10 = [x[-10:] for x in all_top_20]
    # Evaluate coverage of all items
    all_top_1_concated = np.array(all_top_1)
    cov1 = len(np.unique(all_top_1_concated)) / item_num
    all_top_5_concated = np.concatenate(all_top_5)
    cov5 = len(np.unique(all_top_5_concated)) / item_num
    all_top_10_concated = np.concatenate(all_top_10)
    cov10 = len(np.unique(all_top_10_concated)) / item_num
    all_top_20_concated = np.concatenate(all_top_20)
    cov20 = len(np.unique(all_top_20_concated)) / item_num

    # Evaluate coverage of novel items
    with open(data_path + 'less_popular_items.pkl', 'rb') as f:
        less_popular_items = pickle.load(f)
    less_pop_num = len(less_popular_items)
    novel_top_1_concated = np.intersect1d(all_top_1_concated, less_popular_items)
    nov_cov1 = len(novel_top_1_concated) / less_pop_num
    novel_top_5_concated = np.intersect1d(all_top_5_concated, less_popular_items)
    nov_cov5 = len(novel_top_5_concated) / less_pop_num
    novel_top_10_concated = np.intersect1d(all_top_10_concated, less_popular_items)
    nov_cov10 = len(novel_top_10_concated) / less_pop_num
    novel_top_20_concated = np.intersect1d(all_top_20_concated, less_popular_items)
    nov_cov20 = len(novel_top_20_concated) / less_pop_num
    return cov1, cov5, cov10, cov20, nov_cov1, nov_cov5, nov_cov10, nov_cov20


def calculate_div_reward(states, len_states, preds, div_rl_embedding, device):
    total_reward = 0
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(len(states)):
        last_click = states[i][len_states[i] - 1].to(device)
        last_click_emb = div_rl_embedding(last_click)
        top_pred = preds[i].to(device)
        top_pred_emb = div_rl_embedding(top_pred)
        current_reward = 1 - cos_sim(last_click_emb, top_pred_emb)
        total_reward += current_reward
    return total_reward


def calculate_total_nov_reward(top_20, data_path):
    with open(data_path + 'less_popular_items.pkl', 'rb') as f:
        less_popular_items = pickle.load(f)
    is_less_popular_20 = np.in1d(top_20, less_popular_items).reshape(-1, 20)
    is_less_popular_1 = is_less_popular_20[:, -1]
    total_nov_reward = is_less_popular_1.sum()
    return total_nov_reward


def initialize_rel_disc_matrix(size, disc_factor=0.85):
    rel_disc_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            rel_disc_matrix[i, j] = disc_factor**i * disc_factor**(np.max([1, j-i]))
    return torch.tensor(rel_disc_matrix)


def get_novelty_reward_dict(data_path):
    nov_rewards_csv = pd.read_csv(data_path + 'binary_nov_reward.csv', header=None, index_col=0)
    nov_rewards_csv.squeeze()
    nov_rewards_dict = nov_rewards_csv.to_dict()
    print('percentage of positive reward: ', len(nov_rewards_csv[nov_rewards_csv == 1]) / len(nov_rewards_csv))
    return nov_rewards_dict


def set_stdout(results_path, file_name, write_to_file=True):
    if write_to_file:
        print('Outputs are saved to file {}'.format(results_path + file_name))
        sys.stdout = open(results_path + file_name, 'w')
    else:
        pass


def get_stats(data_path):
    data_statis = pd.read_pickle(data_path + 'data_statis.df')  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num

