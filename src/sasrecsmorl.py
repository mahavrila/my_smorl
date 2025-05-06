import time as time
import torch.nn.functional as F
import getpass
from utils import *
from evaluate_stock import evaluate
from SASRecModules import *
import setproctitle
setproctitle.setproctitle('SASRecSMORL')
import os
import random
import numpy as np

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

class SASRecSMORL(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, discount, div_embedding_matrix,
                 smorl_loss_mult, smorl_weights, novelty_rewards_dict, device, num_heads=1):
        super(SASRecSMORL, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.discount = discount
        self.smorl_loss_mult = smorl_loss_mult
        self.novelty_rewards_dict = novelty_rewards_dict
        self.smorl_weights = smorl_weights
        self.device = device
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        nn.init.normal_(self.positional_embeddings.weight, 0, 0.01)

        # Supervised Head Layers
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        # Accuracy RL Head
        self.acc_rl_fc1 = nn.Linear(self.hidden_size, self.item_num)

        # Diversity RL Head
        self.div_rl_fc1 = nn.Linear(self.hidden_size, self.item_num)
        self.div_rl_embedding = div_embedding_matrix

        # Novelty RL Head
        self.nov_rl_fc1 = nn.Linear(self.hidden_size, self.item_num)

    def get_div_rewards(self, states, len_states, preds):
        div_rewards = torch.zeros(len(states)).to(self.device)
        for i in range(len(states)):
            last_click = states[i][len_states[i] - 1]
            last_click_emb = self.div_rl_embedding(last_click)
            top_pred = preds[i]
            top_pred_emb = self.div_rl_embedding(top_pred)
            current_reward = 1 - self.cos_sim(last_click_emb, top_pred_emb)
            div_rewards[i] = current_reward
        return div_rewards

    def get_nov_rewards(self, preds):
        preds = preds.tolist()
        return torch.Tensor([self.novelty_rewards_dict[1][key] for key in preds]).to(self.device)

    def forward(self, states, len_states, actions, acc_rewards, target_Qs, target_Qs_s, is_done):
        inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1).squeeze()
        supervised_output = self.s_fc(state_hidden)
        supervised_preds = torch.argmax(supervised_output, dim=-1)

        # SMORL Head
        acc_q_learning_outputs = self.acc_rl_fc1(state_hidden)
        div_q_learning_output = self.div_rl_fc1(state_hidden)
        nov_q_learning_output = self.nov_rl_fc1(state_hidden)
        smorl_q_learning_output = torch.stack([acc_q_learning_outputs,
                                               div_q_learning_output,
                                               nov_q_learning_output], dim=1)
        smorl_q_values = smorl_q_learning_output.squeeze()[torch.arange(smorl_q_learning_output.shape[0]), :, actions]
        for index in range(target_Qs.shape[0]):
            if is_done[index]:
                target_Qs[index] = torch.zeros([len(self.smorl_weights), self.item_num])
        target_Qs_s = target_Qs_s[:, 0, :] * self.smorl_weights[0] + \
                      target_Qs_s[:, 1, :] * self.smorl_weights[1] + \
                      target_Qs_s[:, 2, :] * self.smorl_weights[2]
        max_actions = torch.argmax(target_Qs_s, dim=-1)
        next_state_smorl_q_values = target_Qs[torch.arange(target_Qs.shape[0]), :, max_actions]
        div_rewards = self.get_div_rewards(states, len_states, supervised_preds)
        nov_rewards = self.get_nov_rewards(supervised_preds)
        rewards = torch.stack([acc_rewards, div_rewards, nov_rewards], dim=0).T
        targets = rewards + self.discount * next_state_smorl_q_values
        smorl_loss = torch.square(targets.detach() - smorl_q_values) * 0.5
        smorl_loss = smorl_loss[:, 0] * self.smorl_weights[0] + \
                     smorl_loss[:, 1] * self.smorl_weights[1] + \
                     smorl_loss[:, 2] * self.smorl_weights[2]
        smorl_loss = smorl_loss.mean()

        return supervised_output, smorl_loss * self.smorl_loss_mult

    def forward_next_states(self, next_states, len_next_states):
        next_states_emb = self.item_embeddings(next_states) * self.item_embeddings.embedding_dim ** 0.5
        next_states_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(next_states_emb)
        mask = torch.ne(next_states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden_next = extract_axis_1(ff_out, len_next_states - 1)

        acc_q_values_next = self.acc_rl_fc1(state_hidden_next)
        div_q_values_next = self.div_rl_fc1(state_hidden_next)
        nov_q_values_next = self.nov_rl_fc1(state_hidden_next)
        next_state_smorl_q_values = torch.stack([acc_q_values_next, div_q_values_next, nov_q_values_next], dim=1)
        return next_state_smorl_q_values.squeeze()

    def forward_eval(self, states, len_states):
        inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1).squeeze()
        supervised_output = self.s_fc(state_hidden)
        acc_q_values = self.acc_rl_fc1(state_hidden)
        div_q_values = self.div_rl_fc1(state_hidden)
        nov_q_values = self.nov_rl_fc1(state_hidden)
        return supervised_output.squeeze(), acc_q_values, div_q_values, nov_q_values


class Args:
    skip=1
    rl=1
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    dataset = 'rc15'
    eval_iter = 5000 if dataset == 'rc15' else 10000
    epochs = 30
    resume = 1
    batch_size = 256
    hidden_factor = 64
    r_click = 0.2
    r_buy = 1.0
    lr = 0.01 if dataset == 'rc15' else 0.005
    dropout_rate = 0.1
    
    cur_dir = os.getcwd()
    root_dir = os.path.dirname(cur_dir)
    div4rec = root_dir + '/div4rec'
    div_emb_matrix = torch.load(f'{div4rec}/{dataset}_models/gru_embedding.pt', weights_only=False)
    div_emb_matrix.weight.requires_grad = False
    data_path_0 = f'{div4rec}/{dataset}_data/Clicks_only/train_skip_{skip}/'
    data_path = f'{div4rec}/{dataset}_data/Clicks_only/'
    models_path = f'{div4rec}/{dataset}_models/'
    results_path = f'{div4rec}/{dataset}_results/SASRec_RL/'
    os.makedirs(results_path, exist_ok=True)
    results_to_file = True  
    
    discount = 0.5
    smorl_weights_set = [torch.Tensor([1.0, 1.0, 1.0])]
    smorl_weights_set_back = [torch.Tensor([1.0, 1.0, 1.0]), torch.Tensor([0.0, 1.0, 0.0]),
                         torch.Tensor([0.0, 0.0, 1.0]), torch.Tensor([0.0, 1.0, 1.0]),
                         torch.Tensor([1.0, 1.0, 0.0]), torch.Tensor([1.0, 0.0, 1.0])]
    smorl_loss_mult = 1
    rel_disc_matrix = initialize_rel_disc_matrix(30)


if __name__ == '__main__':
    writer = SummaryWriter()

    # Network parameters
    args = Args()

    device = set_device(0)
    print('Using {} For Training'.format(torch.cuda.get_device_name(0)))
    for smorl_weights in args.smorl_weights_set:
        file_name = 'sasrec_smorl{}_acc{}_div{}_nov{}_weighted_q_vals.txt'.format(
            str(args.smorl_loss_mult).replace('.', ''),
            smorl_weights[0],
            smorl_weights[1],
            smorl_weights[2])
        set_stdout(args.results_path, file_name, write_to_file=args.results_to_file)
        sys.stdout.flush()
        print("I'm starting sasrec_smorl{}_acc{}_div{}_nov{}_weighted_q_vals".format(
            str(args.smorl_loss_mult).replace('.', ''),
            smorl_weights[0],
            smorl_weights[1],
            smorl_weights[2]
        ))
        nov_rewards_dict = get_novelty_reward_dict(args.data_path)

        state_size, item_num = get_stats(args.data_path)
        train_loader = prepare_dataloader_skips(args.data_path, args.batch_size, skip=args.skip)
        
        sasrec1 = SASRecSMORL(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
            dropout=args.dropout_rate,
            div_embedding_matrix= args.div_emb_matrix,
            discount=args.discount,
            smorl_loss_mult=args.smorl_loss_mult,
            smorl_weights=smorl_weights.to(device),
            novelty_rewards_dict=nov_rewards_dict,
            device=device
        )

        sasrec2 = SASRecSMORL(
            hidden_size=args.hidden_factor,
            item_num=item_num,
            state_size=state_size,
            dropout=args.dropout_rate,
            div_embedding_matrix=args.div_emb_matrix,
            discount=args.discount,
            smorl_loss_mult=args.smorl_loss_mult,
            smorl_weights=smorl_weights.to(device),
            novelty_rewards_dict=nov_rewards_dict,
            device=device
        )

        print("Model:")
        print(sasrec1)
        print("TORCHINFO:")
        summary(sasrec1)
        print("Model ':")
        print(sasrec2)
        print("TORCHINFO:")
        summary(sasrec2)


        criterion = nn.CrossEntropyLoss()
        params1 = list(sasrec1.parameters())
        optimizer1 = torch.optim.Adam(params1, lr=args.lr)

        params2 = list(sasrec2.parameters())
        optimizer2 = torch.optim.Adam(params2, lr=args.lr)

        sasrec1.to(device)
        sasrec2.to(device)

        reward_click = torch.repeat_interleave(torch.Tensor([args.r_click]), args.batch_size)
        reward_buy = torch.repeat_interleave(torch.Tensor([args.r_buy]), args.batch_size)

        # Start training loop
        epoch_times = []
        total_step = 0
        best_val_acc = 6.1418
        
        for epoch in range(0, args.epochs):
            sasrec1.train()
            sasrec2.train()
            start_time = time.time()
            avg_loss = 0.
            
            for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    main_QN = sasrec1
                    target_QN = sasrec2
                    optimizer = optimizer1
                    model_name = 'SASRec SMORL 1'
                else:
                    main_QN = sasrec2
                    target_QN = sasrec1
                    optimizer = optimizer2
                    model_name = 'SASRec SMORL 2'

                main_QN.zero_grad()
                target_QN.zero_grad()

                target_Qs = target_QN.forward_next_states(
                    next_state.to(device),
                    len_next_state.to(device)
                )
                target_Qs_selector = main_QN.forward_next_states(
                    next_state.to(device),
                    len_next_state.to(device)
                )
                acc_reward = torch.where(is_buy > 0, reward_buy, reward_click)
                supervised_out, smorl_loss = main_QN(
                    state.to(device).long(),
                    len_state.to(device).long(),
                    action.to(device),
                    acc_reward.to(device),
                    target_Qs.to(device),
                    target_Qs_selector.to(device),
                    is_done.to(device)
                )
                supervised_loss = criterion(supervised_out, action.to(device).long())
                loss = supervised_loss + smorl_loss
                scalar_loss = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_step += 1
                
                if total_step % 200 == 0:
                    print('Model is ', model_name)
                    print('Supervised loss is %.3f, SMORL loss is %.3f' % (supervised_loss.item(), smorl_loss))
                    print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, scalar_loss))
                    writer.add_scalar(f"Loss/train_{Args.skip}", scalar_loss , total_step)
                    sys.stdout.flush()

                if total_step % args.eval_iter == 0:
                    print('Evaluating Main Model')
                    val_acc_main, predictions_main = evaluate(main_QN, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step, RL=args.rl)
                    val_acc_main, predictions_main = evaluate(main_QN, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step, RL=args.rl)
                    predictions_main.to_csv(
                        '{}pred_main_acc{}_div{}_nov{}_epoch{}_step{}.tsv'.format(args.results_path, smorl_weights[0],
                                                                                  smorl_weights[1], smorl_weights[2],
                                                                                  epoch, total_step), sep='\t',
                        index=False)
                    print("I've saved pred main at {}pred_main_acc{}_div{}_nov{}_epoch{}_step{}.tsv".format(
                        args.results_path, smorl_weights[0], smorl_weights[1], smorl_weights[2], epoch, total_step))
                    main_QN.train()
                    print('Current accuracy of main model: ', val_acc_main)
                    print('Evaluating Target Model')
                    val_acc_target, predictions_target = evaluate(target_QN, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step, RL=args.rl)
                    val_acc_target, predictions_target = evaluate(target_QN, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step, RL=args.rl)
                    predictions_target.to_csv(
                        '{}pred_target_acc{}_div{}_nov{}_epoch{}_step{}.tsv'.format(args.results_path, smorl_weights[0],
                                                                                    smorl_weights[1], smorl_weights[2],
                                                                                    epoch, total_step), sep='\t',
                        index=False)
                    print("I've saved pred tarhet at {}pred_target_acc{}_div{}_nov{}_epoch{}_step{}.tsv".format(
                        args.results_path,
                        smorl_weights[0], smorl_weights[1], smorl_weights[2], epoch, total_step))
                    target_QN.train()
                    print('Current accuracy of target model: ', val_acc_target)

            current_time = time.time()
            print('Epoch {}/{} Done'.format(epoch, args.epochs))
            print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
            epoch_times.append(current_time - start_time)
        writer.flush()
        writer.close()
        print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
        sys.stdout.close()
        sys.stdout = sys.__stdout__