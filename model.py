import torch
from torch import nn
from torch.functional import F
from torch.nn.modules import LayerNorm


import numpy as np
from allennlp.modules.augmented_lstm import AugmentedLstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Average(nn.Module):

    def __init__(self):
        super().__init__()
        # Average vector layer

    def forward(self, sentence):
        # print(sentence)
        return sentence[0].mean(dim=0)

# input dropout
# from pytorchnlp https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.
    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


# attention from Yang 2016
class YangAttnetion(nn.Module):
    def __init__(self, lstm_dim):
        super().__init__()

        self.word_attn = nn.Linear(lstm_dim, lstm_dim)
        self.context_vec = nn.Linear(lstm_dim, 1, bias=False)

    def forward(self, lstm_output):
        # page 1482 top right
        # eq 5, with tanh (from our report)
        u_it = torch.tanh(self.word_attn(lstm_output))
        # eq 6
        a_it = F.softmax(self.context_vec(u_it), dim=1)
        # eq 7
        attns = torch.Tensor()
        for (h, a) in zip(lstm_output, a_it):
            h_i = a*h
            h_i = h_i.unsqueeze(0)
            # add them to the attention vectors
            attns = torch.cat([attns, h_i])

        s_i = torch.sum(attns, 1)
        # unsqueeze to give back to FC layers
        s_i = s_i.unsqueeze(0)

        return s_i

class LSTMEncoder(nn.Module):
    def __init__(self, word_embedding_dim, lstm_dim, bidirectional=False, use_mu_attention=False, use_self_attention=False,use_yang_attention=True, max_pool=False):
        super().__init__()

        self.lstm_dim = lstm_dim
        self.emb_dim = word_embedding_dim
        self.lstm =  AugmentedLstm(word_embedding_dim, lstm_dim, recurrent_dropout_probability=0.5) # nn.LSTM(word_embedding_dim, lstm_dim, 1, bidirectional=bidirectional)
       

        # yang attention
        self.use_yang_attention = use_yang_attention
        if (self.use_yang_attention):
            self.yang_att = YangAttnetion(lstm_dim*2 if bidirectional else lstm_dim)

        self.use_mu_attention = use_mu_attention
        self.use_self_attention = use_self_attention

        self.max_pool = max_pool

    def forward(self, sentence):
        lengths_sorted, sorted_idx = torch.sort(sentence[1], descending=True)
        idx_unsort = torch.argsort(sorted_idx)
        indexed_batch = sentence[0][:, sorted_idx, :]
        packed_sentence = nn.utils.rnn.pack_padded_sequence(
            indexed_batch, lengths_sorted)

        packed_output, (h_n, c_n) = self.lstm(packed_sentence)

        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        if (self.use_yang_attention):
            output = self.yang_att(output)
            
        if (self.use_mu_attention):
            output = self.attention(output, h_n)

        if (self.use_self_attention):
            output = self.self_att(output, output_lengths)

        if not self.max_pool:
            h_n = h_n[:, idx_unsort, :]
            return h_n.permute(1, 0, 2).flatten(1, 2)
        else:
            output = output[:, idx_unsort, :]
            # print(torch.max(output, dim=0))
            return torch.max(output, dim=0)[0]


class TransformerEncoder(nn.Module):
    def __init__(self, dim_model=300, num_heads=12, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            dim_model, num_heads, dim_feedforward, dropout)
        encoder_norm = LayerNorm(dim_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 1, encoder_norm)

    def forward(self, sentence):
        # print(sentence[0].shape)
        output = self.transformer(sentence[0])
        # print(output.shape)
        return torch.max(output, dim=0)[0]


class Main(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.num_embeddings = config['num_embeddings']
        self.embedding_dim = config['embedding_dim']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_classes = config['n_classes']
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False

        self.bidirectional = False
        self.use_yang_attention = True

        if config['encoder'] == "lstm":
            self.input_dim = self.input_dim * config['lstm_dim']
            self.encoder = LSTMEncoder(
                self.embedding_dim, config['lstm_dim'], bidirectional=self.bidirectional, use_yang_attention=self.use_yang_attention)
        if config['encoder'] == "average":
            self.input_dim = self.input_dim * 300
            self.encoder = Average()
        if config['encoder'] == "transformer":
            self.input_dim = self.input_dim * 300
            self.encoder = TransformerEncoder()
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(
                self.input_dim * 2 if self.bidirectional else self.input_dim, self.n_classes),
        )

    def forward_encoder(self, sentence):
        s1 = self.embedding(sentence[0])
        u = self.encoder((s1, sentence[1]))
        return u

    def forward(self, text):
        s1 = self.embedding(text[0])
        s1_dropped = LockedDropout(p=0.2)(s1)
        u = self.encoder((s1_dropped, text[1]))
        features = u
        # print(features.shape)
        return self.classifier(features)
