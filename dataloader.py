from torchtext import data
from torch import nn

train, test = data.TabularDataset.splits(
path='./dataset/', train='Olid-training-v1.0.tsv', test='labels-levela.tsv', format='tsv',
fields=[('tweet', data.Field()),
        ('subtask_a', data.Field())])
        
TEXT = data.Field()

TEXT.build_vocab(train, vectors='GLOVE')
vocab = TEXT.vocab
embed = nn.Embedding(len(vocab), 200)
embed.weight.data.copy_(vocab.vectors)


print (TEXT)