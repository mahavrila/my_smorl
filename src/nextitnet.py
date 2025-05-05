import time
import torch.nn as nn
from utils import *
from evaluate_stock import evaluate
from NextItNetModules import ResidualBlock
import setproctitle

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

setproctitle.setproctitle('NextItNet')


class NextItNet(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dilations, device):
        super(NextItNet, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.dilations = eval(dilations)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # Initialize embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Convolutional Layers
        self.cnns = nn.ModuleList([
            ResidualBlock(
                in_channels=1,
                residual_channels=hidden_size,
                kernel_size=3,
                dilation=i,
                hidden_size=hidden_size) for i in self.dilations
        ])
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(device)
        emb *= mask
        conv_out = emb
        for cnn in self.cnns:
            conv_out = cnn(conv_out)
            conv_out *= mask
        state_hidden = extract_axis_1(conv_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(device)
        emb *= mask
        conv_out = emb
        for cnn in self.cnns:
            conv_out = cnn(conv_out)
            conv_out *= mask
        state_hidden = extract_axis_1(conv_out, len_states - 1)
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
    
    cur_dir = os.getcwd()
    root_dir = os.path.dirname(cur_dir)
    div4rec = root_dir + '/div4rec'
    div_emb_matrix = torch.load(f'{div4rec}/{dataset}_models/gru_embedding.pt', weights_only=False)
    div_emb_matrix.weight.requires_grad = False
    data_path_0 = f'{div4rec}/{dataset}_data/Clicks_only/train_skip_1/'
    data_path = f'{div4rec}/{dataset}_data/Clicks_only/'
    models_path = f'{div4rec}/{dataset}_models/'
    results_path = f'{div4rec}/{dataset}_results/NextItNet/'
    os.makedirs(results_path, exist_ok=True)
    results_to_file = True
    
    num_filters = 16
    dilations = '[1, 2, 1, 2, 1, 2]'


if __name__ == '__main__':
    writer = SummaryWriter()
    args = Args()
    device = set_device(0)
    print('Using {} For Training'.format(torch.cuda.get_device_name()))
    file_name = 'nextitnet.txt'
    set_stdout(args.results_path, file_name, write_to_file=args.results_to_file)
    sys.stdout.flush()

    state_size, item_num = get_stats(args.data_path)
    train_loader = prepare_dataloader_skips(args.data_path, args.batch_size, skip=Args.skip)

    nextItNet = NextItNet(
        hidden_size=args.hidden_factor,
        item_num=item_num,
        state_size=state_size,
        dilations=args.dilations,
        device=device
    )

    print(nextItNet)
    print("TORCHINFO:")
    summary(nextItNet)
    
    criterion = nn.CrossEntropyLoss()
    params1 = list(nextItNet.parameters())
    optimizer = torch.optim.Adam(params1, lr=args.lr)

    nextItNet.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 6.1418
    
    for epoch in range(0, args.epochs):
        nextItNet.train()
        start_time = time.time()
        avg_loss = 0.

        for state, len_state, action, next_state, len_next_state, is_buy, is_done in train_loader:
            nextItNet.zero_grad()
            supervised_out = nextItNet(state.to(device).long(), len_state.long())
            supervised_loss = criterion(supervised_out, action.to(device).long())
            scalar_loss = supervised_loss.item()
            optimizer.zero_grad()
            supervised_loss.backward()
            optimizer.step()
            total_step += 1
            
            if total_step % 200 == 0:
                print('Model is NextItNet',)
                print('Supervised loss is %.3f' % (supervised_loss.item()))
                print('Epoch {}......Step: {}....... Loss: {}'.format(epoch, total_step, scalar_loss))
                writer.add_scalar(f"Loss/train_{Args.skip}", scalar_loss, total_step)
                sys.stdout.flush()

            if total_step % args.eval_iter == 0:
                print('Evaluating Vanilla NextItNet Model')
                val_acc, predictions = evaluate(nextItNet, args, 'val', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                val_acc, predictions = evaluate(nextItNet, args, 'test', state_size, item_num, device, args.div_emb_matrix, skip=args.skip, writer=writer, step=total_step)
                predictions.to_csv(
                    '{}pred_nextitnet_epoch{}_step{}.tsv'.format(args.results_path, epoch, total_step), sep='\t', index=False)
                print(
                    "I've saved pred main at {}pred_nextitnet_epoch{}_step{}.tsv".format(args.results_path, epoch,
                                                                                   total_step))
                nextItNet.train()
                print('Current accuracy: ', val_acc)
        current_time = time.time()
        print('Epoch {}/{} Done'.format(epoch, args.epochs))
        print('Total Time Elapsed: {} seconds'.format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    writer.flush()
    writer.close()
    print('Total Training Time: {} seconds'.format(str(sum(epoch_times))))
    sys.stdout.close()