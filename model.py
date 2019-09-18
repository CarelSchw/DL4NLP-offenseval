import torch
from torch import nn

from torch.nn.modules import LayerNorm


import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Average(nn.Module):

    def __init__(self):
        super().__init__()
        # Average vector layer

    def forward(self, sentence):
        # print(sentence)
        return sentence[0].mean(dim=0)


# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.device = device

        self.att_weights = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=True)

        # Vaswani et al 2017
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, lengths):
        batch_size, max_len = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            # (batch_size, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            )

        attentions = torch.softmax(
            torch.nn.functional.relu(weights.squeeze()), dim=-1)

       # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(),
                          requires_grad=True).to(device)
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class LSTMEncoder(nn.Module):
    def __init__(self, word_embedding_dim, lstm_dim, bidirectional=False, use_mu_attention=False, use_self_attention=False, max_pool=False):
        super().__init__()

        self.lstm = nn.LSTM(word_embedding_dim, lstm_dim, 1,
                            bidirectional=bidirectional)
        self.self_att = Attention(
            lstm_dim*2 if bidirectional else lstm_dim)  # 2 is bidrectional

        self.use_mu_attention = use_mu_attention
        self.use_self_attention = use_self_attention

        self.max_pool = max_pool

    # TODO: check if this is correct multiplicative attention
    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, sentence):
        lengths_sorted, sorted_idx = torch.sort(sentence[1], descending=True)
        idx_unsort = torch.argsort(sorted_idx)
        indexed_batch = sentence[0][:, sorted_idx, :]
        packed_sentence = nn.utils.rnn.pack_padded_sequence(
            indexed_batch, lengths_sorted)

        packed_output, (h_n, c_n) = self.lstm(packed_sentence)

        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

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
    def __init__(self, dim_model=300, num_heads=6, dim_feedforward=2048, dropout=0.1):
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
        self.use_self_attention = False

        if config['encoder'] == "lstm":
            self.input_dim = self.input_dim * config['lstm_dim']
            self.encoder = LSTMEncoder(
                self.embedding_dim, config['lstm_dim'], bidirectional=self.bidirectional, use_self_attention=self.use_self_attention)
        if config['encoder'] == "average":
            self.input_dim = self.input_dim * 150
            self.encoder = Average()
        if config['encoder'] == "transformer":
            self.input_dim = self.input_dim * 150
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
        u = self.encoder((s1, text[1]))
        features = u
        # print(features.shape)
        return self.classifier(features)
