import torch
from torch import nn

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Average(nn.Module):

    def __init__(self):
        super().__init__()
        # Average vector layer

    def forward(self, sentence):
        # print(sentence)
        return sentence[0].mean(dim=0)


class LSTMEncoder(nn.Module):
    def __init__(self, word_embedding_dim, lstm_dim, bidirectional=False, max_pool=False):
        super().__init__()

        self.lstm = nn.LSTM(word_embedding_dim, lstm_dim, 1,
                            bidirectional=bidirectional)
        self.max_pool = max_pool

    def forward(self, sentence):
        lengths_sorted, sorted_idx = torch.sort(sentence[1], descending=True)
        idx_unsort = torch.argsort(sorted_idx)
        indexed_batch = sentence[0][:, sorted_idx, :]
        packed_sentence = nn.utils.rnn.pack_padded_sequence(
            indexed_batch, lengths_sorted)
        packed_output, (h_n, c_n) = self.lstm(packed_sentence)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output)
        if not self.max_pool:
            h_n = h_n[:, idx_unsort, :]
            return h_n.permute(1, 0, 2).flatten(1, 2)
        else:
            output = output[:, idx_unsort, :]
            # print(torch.max(output, dim=0))
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
        if config['encoder'] == "lstm":
            self.input_dim = self.input_dim * config['lstm_dim']
            self.encoder = LSTMEncoder(
                self.embedding_dim, config['lstm_dim'], bidirectional=False)
        if config['encoder'] == "average":
            self.input_dim = self.input_dim * 300
            self.encoder = Average()
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(self.input_dim, self.n_classes),
        )

    def forward_encoder(self, sentence):
        s1 = self.embedding(sentence[0])
        u = self.encoder((s1, sentence[1]))
        return u

    def forward(self, text):
        s1 = self.embedding(text[0])
        u = self.encoder((s1, text[1]))
        features = u
        return self.classifier(features)
