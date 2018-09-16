import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
from lstm import LSTM
from gru import GRU
from exrnn import EXRNN
from rnn import RNN
from ann import ANN
from wann import WANN


def run_model(which='all'):
    if which in ['ann', 'all', 'main', 'standard']:
        model = ANN(emb_size, vocab_size, hid_dim, hid_num, class_num, sent_len).cuda()
        ann_loss = train(model, x, target, ann=True)
        plt.plot(ann_loss, label='ann')
    if which in ['wann', 'all', 'standard']:
        model = WANN(emb_size, vocab_size, hid_dim, hid_num, class_num, sent_len).cuda()
        wann_loss = train(model, x, target, ann=True)
        plt.plot(wann_loss, label='wann')
    if which in ['rnn', 'all', 'main']:
        model = RNN(emb_size, vocab_size, hid_dim, hid_num, class_num).cuda()
        rnn_loss = train(model, x, target)
        plt.plot(rnn_loss, label='rnn')
    if which in ['exrnn', 'all']:
        model = EXRNN(emb_size, vocab_size, hid_dim, hid_num,
                      class_num, 2000, 2000).cuda()
        exrnn_loss = train(model, x, target)
        plt.plot(exrnn_loss, label='exrnn')
    if which in ['exmem', 'all']:
        model = EXRNN(emb_size, vocab_size, hid_dim, hid_num,
                      class_num, 2000, forget_dim=None).cuda()
        exmem_loss = train(model, x, target)
        plt.plot(exmem_loss, label='exmem')
    if which in ['lstm', 'all', 'main']:
        model = LSTM(emb_size, vocab_size, hid_dim, hid_num, class_num).cuda()
        lstm_loss = train(model, x, target)
        plt.plot(lstm_loss, label='lstm')
    if which in ['gru', 'all', 'main']:
        model = GRU(emb_size, vocab_size, hid_dim,
                    hid_num, class_num).cuda()
        gru_loss = train(model, x, target)
        plt.plot(gru_loss, label='gru')
    # plt.ylim([0, 2])
    plt.legend()
    plt.grid(True)
    plt.show()


def train(model, x, target, forget=True, ann=False):
    if torch.cuda.is_available():
        x = x.cuda()
        target = target.cuda()
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    ls = []
    for i in range(epochs):
        model.zero_grad()
        if ann:
            y_pred = model(x.cuda())
        else:
            if forget is False:
                y_pred, all_y = model(x.cuda(), forget)
            else:
                y_pred, all_y = model(x.cuda())
        loss = criterion(y_pred, target.cuda())
        loss.backward()
        optimizer.step()
        print('epoch {}, loss:{:.8f}'.format(i + 1, loss.item()))
        ls.append(loss.data[0])
    return ls


if __name__ == "__main__":

    value_mem = True
    vocab_size = 50
    emb_size = 100
    hid_dim = emb_size
    hid_num = 1
    class_num = 10
    classes = [1, 2, 5, 8, 10, 15, 20]
    batch_size = 100
    sent_len = 30
    epochs = 40

    x = Variable(torch.randint(0, vocab_size - 1, (sent_len, batch_size)).type(torch.LongTensor))
    if value_mem:
        target = Variable(x[0, :])
    else:
        target = Variable(torch.randint(0, class_num, (batch_size,)).type(torch.LongTensor))
    run_model('main')
