import torch
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, emb_size, vocab_size, hid_dim, hid_num, class_num):
        super(RNN, self).__init__()

        self.emb_size = emb_size
        self.hid_dim = hid_dim
        self.hid_num = hid_num
        self.nclass = class_num
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.hweight = []
        self.hbias = []

        assert emb_size == hid_dim
        self.h0 = torch.zeros((1, hid_dim)).type(torch.FloatTensor).cuda()

        for hid in range(hid_num):
            if hid == 0:
                # 14 because c_tilde, i, f, c and 2 each for hidden and input
                w1 = torch.normal(torch.randn(hid_dim, hid_dim * 6))*0.01
                b1 = torch.zeros(1, hid_dim * 6)
                self.w1 = nn.Parameter(w1, requires_grad=True)
                self.b1 = nn.Parameter(b1, requires_grad=True)
                self.hweight.append(self.w1)
                self.hbias.append(self.b1)
            else:
                self.hweight.append(nn.Parameter(torch.randn(hid_dim, 2* hid_dim)))
                self.hbias.append(nn.Parameter(torch.zeros(1, hid_dim)))
        self.fc1 = nn.Linear(hid_dim, class_num)

    def forward_step(self, x, htm1, lnum=0):
        x_in = torch.mm(x, self.hweight[lnum][:, :self.hid_dim]) + self.hbias[lnum][:, :self.hid_dim] \
               + torch.mm(htm1, self.hweight[lnum][:, self.hid_dim: 2 * self.hid_dim])
        h_t = F.tanh(x_in)
        return h_t

    # (sent_len, batch_size, embedding_size)
    def forward(self, x, h_t=None, all_out=True):
        x = self.emb(x)
        if all_out:
            out = []
        for t in range(x.size(0)):
            for layer in range(self.hid_num):
                x_in = x[t, :, :]
                if t == 0:
                    h_t = self.forward_step(x_in, self.h0)
                else:
                    h_t = self.forward_step(x_in, h_t)
            out.append(h_t)
        out = torch.cat(out, 1)
        # print(h_t)
        y = self.fc1(h_t)
        return y, out

