import time
from src.utils import *
from src.evaluate_stock import evaluate
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

import setproctitle
setproctitle.setproctitle('GRU')


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
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
    
    cur_dir = os.getcwd()
    root_dir = os.path.dirname(cur_dir)
    div4rec = root_dir + '/div4rec'
    div_emb_matrix = torch.load(f'{div4rec}/{dataset}_models/gru_embedding.pt', weights_only=False)
    div_emb_matrix.weight.requires_grad = False
    data_path_0 = f'{div4rec}/{dataset}_data/Clicks_only/train_skip_{skip}/'
    data_path = f'{div4rec}/{dataset}_data/Clicks_only/'
    models_path = f'{div4rec}/{dataset}_models/'
    results_path = f'{div4rec}/{dataset}_results/gru_vanilla/'
    os.makedirs(results_path, exist_ok=True)
    results_to_file = True


if __name__ == '__main__':
    writer = SummaryWriter()
    args = Args()
    device = set_device(0)
    print('Using {} For Training'.format(torch.cuda.get_device_name()))
    file_name = 'gru.txt'
    set_stdout(args.results_path, file_name, write_to_file=args.results_to_file)
    sys.stdout.flush()

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader_skips(args.data_path, args.batch_size, skip=args.skip)

    # initialize model, print model info
    gru = GRU(hidden_size=args.hidden_factor,           # 64
              item_num=item_num,                        # 26702
              state_size=state_size,                    # 10
              )
    print(gru)
    print("TORCHINFO:")
    summary(gru)
    
    # add loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(gru.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # move model to GPU
    gru.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 7.5

    for epoch in range(0, args.epochs):
        start_time = time.time()
        avg_loss = 0.
        
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            gru.zero_grad()
            supervised_out = gru(state.to(device).long(), len_state.long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            scalar_loss = supervised_loss.item()
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()
            total_step += 1

            if total_step % 200 == 0:
                print('Model is Vanilla GRU')
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, scalar_loss))
                writer.add_scalar(f"Loss/train_{Args.skip}", scalar_loss, total_step)
                sys.stdout.flush()

            if total_step % args.eval_iter == 0:
                print('Evaluating Vanilla GRU Model')
                # val_acc = evaluate(gru, args, 'val', state_size, item_num, device, args.div_emb_matrix)
                val_acc, predictions = (evaluate(gru, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=1, writer=writer, step=total_step))
                val_acc, predictions = (evaluate(gru, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=1, writer=writer, step=total_step))
                predictions.to_csv(
                    '{}pred_gru_epoch{}_step{}.tsv'.format(args.results_path, epoch, total_step),
                    sep='\t', index=False)
                print(
                    "I've saved pred main at {}pred_gru_epoch{}_step{}.tsv".format(
                        args.results_path, epoch, total_step))
                gru.train()
                print('Current accuracy: ', val_acc)
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    writer.flush()
    writer.close()
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()      