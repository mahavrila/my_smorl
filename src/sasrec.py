import torch.nn as nn
import pickle
import time
from utils import *
from evaluate_stock import evaluate
from SASRecModules import *
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import setproctitle
setproctitle.setproctitle('SASRec')


class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
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
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

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
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


class Args:
    skip=1
    dataset = 'rc15'
    eval_iter = 5000 if dataset == 'rc15' else 10000
    epochs = 30
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
    data_path_0 = f'{div4rec}/{dataset}_data/Clicks_only/train_skip_1/'
    data_path = f'{div4rec}/{dataset}_data/Clicks_only/'
    models_path = f'{div4rec}/{dataset}_models/'
    results_path = f'{div4rec}/{dataset}_results/SASRec/'
    os.makedirs(results_path, exist_ok=True)
    results_to_file = True


if __name__ == '__main__':
    writer = SummaryWriter()
    args = Args()
    device = set_device(0)
    print('Using {} For Training'.format(torch.cuda.get_device_name()))
    file_name = 'sasrec.txt'
    set_stdout(args.results_path, file_name, write_to_file=args.results_to_file)
    sys.stdout.flush()

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader_skips(args.data_path, args.batch_size, skip=Args.skip)

    sasRec = SASRec(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        dropout=args.dropout_rate,
        device=device
    )
    
    print(sasRec)
    print("TORCHINFO:")
    summary(sasRec)
    
    criterion = nn.CrossEntropyLoss()
    params1 = list(sasRec.parameters())
    optimizer = torch.optim.Adam(params1, lr=args.lr)

    sasRec.to(device)

    reward_click = args.r_click
    reward_buy = args.r_buy

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 6.1418
    
    for epoch in range(0, args.epochs):
        sasRec.train()
        start_time = time.time()
        avg_loss = 0.

        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            sasRec.zero_grad()
            supervised_out = sasRec(state.to(device).long(), len_state.long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            scalar_loss = supervised_loss.item()
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()
            total_step += 1
                         
            if total_step % 200 == 0:
                print('Model is Vanilla SASRec', )
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, scalar_loss))
                writer.add_scalar(f"Loss/train_{Args.skip}", scalar_loss, total_step)
                sys.stdout.flush()

            if total_step % args.eval_iter == 0:
                print('Evaluating Vanilla SASREC Model')
                # val_acc = evaluate(sasRec, args, 'val', state_size, item_num, device, args.div_emb_matrix)
                val_acc, predictions = evaluate(sasRec, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                val_acc, predictions = evaluate(sasRec, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                predictions.to_csv(
                    '{}pred_sasrec_epoch{}_step{}.tsv'.format(args.results_path, epoch, total_step), sep='\t', index=False)
                print(
                    "I've saved pred main at {}pred_sasrec_epoch{}_step{}.tsv".format(args.results_path, epoch,
                                                                                   total_step))
                sys.stdout.flush()
                sasRec.train()
                print('Current accuracy: ', val_acc)
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    writer.flush()
    writer.close()
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()