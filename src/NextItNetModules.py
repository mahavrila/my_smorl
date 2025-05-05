import torch
import torch.nn as nn
import torch.nn.functional as F


class VerticalCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, hidden_size):
        super(VerticalCausalConv, self).__init__()

        # attributes:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation
        self.hidden_size = hidden_size
        assert out_channels == hidden_size

        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, hidden_size),
            dilation=(dilation, 1)
        )

    def forward(self, seq):
        seq = F.pad(seq, pad=[0, 0, (self.kernel_size - 1) * self.dilation, 0])
        conv2d_out = self.conv2d(seq)
        return conv2d_out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, residual_channels, kernel_size, dilation, hidden_size):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hidden_size = hidden_size
        self.residual_channels = residual_channels
        assert residual_channels == hidden_size  # In order for output to be the same size

        self.conv1 = VerticalCausalConv(
            in_channels=in_channels,
            out_channels=residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            hidden_size=hidden_size
        )
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.conv2 = VerticalCausalConv(
            in_channels=in_channels,
            out_channels=residual_channels,
            kernel_size=kernel_size,
            dilation=dilation * 2,
            hidden_size=hidden_size
        )
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, input_):
        input_unsqueezed = input_.unsqueeze(1)
        conv1_out = self.conv1(input_unsqueezed).permute(0, 3, 2, 1)

        ln1_out = self.ln1(conv1_out)
        relu1_out = F.relu(ln1_out)

        conv2_out = self.conv2(relu1_out).permute(0, 3, 2, 1)
        ln2_out = self.ln2(conv2_out)
        relu2_out = F.relu(ln2_out)
        relu2_out = relu2_out.squeeze()

        out = input_ + relu2_out
        return out