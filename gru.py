import torch
from torch import nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, emb_size, vocab_size, hid_dim, hid_num, class_num):
        super(GRU, self).__init__()

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
                self.hweight.append(nn.Parameter(torch.randn(hid_dim, hid_dim * 6)))
                self.hbias.append(nn.Parameter(torch.zeros(1, hid_dim * 6)))
        self.fc1 = nn.Linear(hid_dim, class_num)

    def forward_step(self, x, htm1, lnum=0):
        x_in = torch.mm(x, self.hweight[lnum][:, :self.hid_dim]) \
               + self.hbias[lnum][:, :self.hid_dim] \
               + torch.mm(htm1, self.hweight[lnum][:, self.hid_dim:2*self.hid_dim]) \
               + self.hbias[lnum][:, self.hid_dim:2 * self.hid_dim]
        r_t = F.sigmoid(x_in)

        x_in = torch.mm(x, self.hweight[lnum][:, 2 * self.hid_dim: 3 * self.hid_dim]) \
            + self.hbias[lnum][:, 2 * self.hid_dim:3 * self.hid_dim] \
            + torch.mm(htm1, self.hweight[lnum][:, 3 * self.hid_dim: 4 * self.hid_dim]) \
            + self.hbias[lnum][:, 3 * self.hid_dim:4 * self.hid_dim]
        z_t = F.sigmoid(x_in)

        x_in = torch.mm(x, self.hweight[lnum][:, 4 * self.hid_dim: 5 * self.hid_dim]) \
            + self.hbias[lnum][:, 4 * self.hid_dim: 5 * self.hid_dim] \
            + r_t * (torch.mm(htm1, self.hweight[lnum][:, 5 * self.hid_dim: 6 * self.hid_dim]) \
            + (self.hbias[lnum][:, 5 * self.hid_dim: 6 * self.hid_dim]))
        n_t = F.sigmoid(x_in)
        h_t = (1 - z_t) * n_t + z_t * htm1
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

