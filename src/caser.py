import torch.nn as nn
import time
from utils import *
from evaluate_stock import evaluate
import setproctitle
setproctitle.setproctitle('Caser')

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)

        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output


class Args:
    skip = 1
    dataset = 'rc15'
    eval_iter = 5000 if dataset == 'rc15' else 10000
    epochs = 30
    batch_size = 256
    hidden_factor = 64
    r_click = 0.2
    r_buy = 1.0
    lr = 0.01 if dataset == 'rc15' else 0.005
    num_filters = 16
    filter_sizes = '[2,3,4]'
        
    cur_dir = os.getcwd()
    root_dir = os.path.dirname(cur_dir)
    div4rec = root_dir + '/div4rec'
    div_emb_matrix = torch.load(f'{div4rec}/{dataset}_models/gru_embedding.pt', weights_only=False)
    div_emb_matrix.weight.requires_grad = False
    data_path_0 = f'{div4rec}/{dataset}_data/Clicks_only/train_skip_{skip}/'
    data_path = f'{div4rec}/{dataset}_data/Clicks_only/'
    models_path = f'{div4rec}/{dataset}_models/'
    results_path = f'{div4rec}/{dataset}_results/Caser_vanilla_sk{skip}/'
    os.makedirs(results_path, exist_ok=True)
    results_to_file = True
    dropout_rate = 0.1


if __name__ == '__main__':
    writer = SummaryWriter()
    args = Args()
    device = set_device(0)
    print('Using {} For Training'.format(torch.cuda.get_device_name()))
    file_name = 'caser.txt'
    set_stdout(args.results_path, file_name, write_to_file=args.results_to_file)
    sys.stdout.flush()

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader_skips(args.data_path, args.batch_size, skip=Args.skip)

    caser = Caser(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        dropout_rate=args.dropout_rate
    )
    print(caser)
    print("TORCHINFO:")
    summary(caser)
    
    criterion = nn.CrossEntropyLoss()
    params = list(caser.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    caser.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 6.1418
    
    for epoch in range(0, args.epochs):
        caser.train()
        start_time = time.time()
        avg_loss = 0.
        
        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            caser.zero_grad()
            supervised_out = caser(state.to(device).long(), len_state.long()) # len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            scalar_loss = supervised_loss.item()

            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()
            total_step += 1

            if total_step % 200 == 0:
                print('Model is Vanilla Caser')
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, scalar_loss))
                writer.add_scalar(f"Loss/train_{Args.skip}", scalar_loss, total_step)
                sys.stdout.flush()

            if total_step % args.eval_iter == 0:
                print('Evaluating Vanilla Caser Model')
                val_acc, predictions = evaluate(caser, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                val_acc, predictions = evaluate(caser, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                predictions.to_csv(
                    '{}pred_gru_epoch{}_step{}.tsv'.format(args.results_path, epoch, total_step), sep='\t', index=False)
                print(
                    "I've saved pred main at {}pred_caser_epoch{}_step{}.tsv".format(args.results_path, epoch,
                                                                                   total_step))
                caser.train()
                print('Current accuracy: ', val_acc)
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    writer.flush()
    writer.close()
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()