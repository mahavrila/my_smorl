import time
from utils import *
import pandas as pd


def evaluate(model, args, val_or_test, state_size, item_num, device, div_rl_embedding, skip=1, writer=None, step=1, RL=0):
    start_time = time.time()
    topk = [5, 10, 20]
    reward_click = args.r_click
    reward_buy = args.r_buy
    if val_or_test == "val":
        eval_sessions = pd.read_pickle(args.data_path + f'sampled_val.df')
    else:
        eval_sessions = pd.read_pickle(args.data_path + f'sampled_test.df')
    sess_ids = eval_sessions.session_id
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_div_reward = 0.0
    total_reward = [0, 0, 0]
    hit_clicks = [0, 0, 0]
    ndcg_clicks = [0, 0, 0]
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    all_top_20 = []
    all_ids = []
    model.to(device)
    model.eval()
    while evaluated < len(eval_ids):                                # this iterates sessions
        ids, states, len_states, actions, rewards = [], [], [], [], []
        for i in range(batch):
            if evaluated == len(eval_ids):                          # just to break out of loop at the end
                break
            id = eval_ids[evaluated]                                # this practically iterates over session ids
            group = groups.get_group(id)                            # here we get single session
            history = []
            counter=1
            for index, row in group.iterrows():                     # iterate through group
                state = list(history)
                temp_length = state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state)
                state = pad_history(state, state_size, item_num)
                s_id = row['session_id']
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if counter > skip:
                    if is_buy == 1:
                        total_purchase += 1.0
                    else:
                        total_clicks += 1.0
                    ids.append(s_id)
                    actions.append(action)
                    rewards.append(reward)
                    states.append(state)
                    len_states.append(temp_length)
                history.append(row['item_id'])
                counter+=1
            evaluated += 1
        states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
        len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
        if RL == 0:
            preds = model.forward_eval(states.to(device).long(), len_states.cpu().long())
        else:
            preds, _, _, _ = model.forward_eval(states.to(device).long(), len_states.cpu().long())
        sorted_list = np.argsort(preds.tolist())

        # Evaluate accuracy measurements (NDCGs & HRs)
        # sorted_list, topk, actions, rewards, reward_click are inputs
        # total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase are to be calculated
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward,
                      hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)

        # Calculate diversity reward
        top_preds = preds.argmax(dim=-1).to(device)
        div_reward = calculate_div_reward(states, len_states, top_preds, div_rl_embedding, device)
        total_div_reward += div_reward

        top_20 = sorted_list[:, -20:].tolist()
        all_top_20 += top_20

        all_ids.extend(ids)

        del states
        del len_states
        torch.cuda.empty_cache()

    cov1, cov5, cov10, cov20, nov_cov1, nov_cov5, nov_cov10, nov_cov20 = eval_cov_nov_cov(
        all_top_20,
        item_num,
        args.data_path
    )
    sess_avg_num_rpts_5, sess_avg_num_rpts_10, sess_avg_num_rpts_20 = calculate_session_repetitions(
        all_ids,
        all_top_20
    )

    all_top_20_numpy = np.concatenate(all_top_20, axis=0)
    total_nov_reward = calculate_total_nov_reward(all_top_20_numpy, args.data_path)

    val_acc = 0
    if val_or_test == "val":
        for i in range(len(topk)):
            hr_click = hit_clicks[i] / total_clicks
            hr_purchase = hit_purchase[i] #/ total_purchase
            ng_click = ndcg_clicks[i] / total_clicks
            ng_purchase = ndcg_purchase[i] #/ total_purchase
            val_acc = val_acc + hr_click + hr_purchase + ng_click + ng_purchase
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
            print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
            writer.add_scalar(f"HR_{skip}/{topk[i]}", hr_click,  step)
            print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
        print('#########################################################################')
        print('total diversity reward: %f' % total_div_reward)
        print('total novelty reward: %f' % total_nov_reward)
        print('#########################################################################')
        print('coverage of top 1 predictions: %f' % cov1)
        writer.add_scalar("COV/1", cov1, step)
        print('coverage of top 5 predictions: %f' % cov5)
        writer.add_scalar("COV/5", cov5, step)
        print('coverage of top 10 predictions: %f' % cov10)
        writer.add_scalar("COV/10", cov10, step)
        print('coverage of top 20 predictions: %f' % cov20)
        writer.add_scalar("COV/20", cov20, step)
        print('#########################################################################')
        print('coverage on novel items of top 1 predictions: %f' % nov_cov1)
        writer.add_scalar("NOV/1", nov_cov1, step)
        print('coverage on novel items of top 5 predictions: %f' % nov_cov5)
        writer.add_scalar("NOV/5", nov_cov5, step)
        print('coverage on novel items of top 10 predictions: %f' % nov_cov10)
        writer.add_scalar("NOV/10", nov_cov10, step)
        print('coverage on novel items of top 20 predictions: %f' % nov_cov20)
        writer.add_scalar("NOV/20", nov_cov20, step)
        print('#########################################################################')
        print(f'average number of repetitions in top 5: {sess_avg_num_rpts_5}')
        writer.add_scalar(f"REP/5_5", sess_avg_num_rpts_5[5], step)
        #writer.add_scalar(f"REP/5_11", sess_avg_num_rpts_5[10])
        print(f'average number of repetitions in top 10: {sess_avg_num_rpts_10}')
        writer.add_scalar(f"REP/10_5", sess_avg_num_rpts_10[5], step)
        #writer.add_scalar(f"REP/10_11", sess_avg_num_rpts_10[10])
        print(f'average number of repetitions in top 20: {sess_avg_num_rpts_20}')
        writer.add_scalar(f"REP/20_5", sess_avg_num_rpts_20[5], step)
        #writer.add_scalar(f"REP/20_11", sess_avg_num_rpts_20[10])
        print('#########################################################################')
        print('total time needed for the evaluation : ', time.time() - start_time)
        print('#########################################################################')
    else:
        for i in range(len(topk)):
            hr_click = hit_clicks[i] / total_clicks
            hr_purchase = hit_purchase[i] #/ total_purchase
            ng_click = ndcg_clicks[i] / total_clicks
            ng_purchase = ndcg_purchase[i] #/ total_purchase
            val_acc = val_acc + hr_click + hr_purchase + ng_click + ng_purchase
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('TEST cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
            print('TEST clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
            print('TEST purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
        print('#########################################################################')
        print('TEST total diversity reward: %f' % total_div_reward)
        print('TEST total novelty reward: %f' % total_nov_reward)
        print('#########################################################################')
        print('TEST coverage of top 1 predictions: %f' % cov1)

        print('TEST coverage of top 5 predictions: %f' % cov5)
        print('TEST coverage of top 10 predictions: %f' % cov10)
        print('TEST coverage of top 20 predictions: %f' % cov20)
        print('#########################################################################')
        print('TEST coverage on novel items of top 1 predictions: %f' % nov_cov1)
        print('TEST coverage on novel items of top 5 predictions: %f' % nov_cov5)
        print('TEST coverage on novel items of top 10 predictions: %f' % nov_cov10)
        print('TEST coverage on novel items of top 20 predictions: %f' % nov_cov20)
        print('#########################################################################')
        print(f'TEST average number of repetitions in top 5: {sess_avg_num_rpts_5}')
        print(f'TEST average number of repetitions in top 10: {sess_avg_num_rpts_10}')
        print(f'TEST average number of repetitions in top 20: {sess_avg_num_rpts_20}')
        print('#########################################################################')
        print('TEST total time needed for the evaluation : ', time.time() - start_time)
        print('#########################################################################')
    return val_acc, pd.DataFrame(all_top_20)

