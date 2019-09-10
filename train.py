import argparse
import os

from torch import nn
from torch import optim
import torch

import data
from model import Main


LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.99
LR_DIVISION = 5
MINIBATCH_SIZE = 64
THRESHOLD = 10 ** -5
HIDDEN_LAYER_UNITS = 512
N_CLASSES = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--datadir", type=str, default='dataset'),
parser.add_argument("--outputdir", type=str,
                    default='savedir/', help="Output directory")
parser.add_argument("--model", type=str, default='average')

params, _ = parser.parse_known_args()

# writer = SummaryWriter(params.outputdir + 'runs/' + params.model)

model_config = {
    'num_embeddings': 1000,
    'embedding_dim': 300,
    'input_dim': 1,
    'hidden_dim': HIDDEN_LAYER_UNITS,
    'n_classes': N_CLASSES,
    'lstm_dim': 2048,
    'encoder': params.model,
}


def accuracy(predictions, targets):
    correct = torch.sum(torch.diag(torch.mm(predictions,
                                            targets.transpose(1, 0))).round())

    denom = targets.shape[0] - (predictions == 0).sum(dim=1)
    accuracy = correct / denom
    return accuracy


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # get batch iterator
    train_set, val_set, test_set = data.preprocess_data(
        data_folder=params.datadir)

    print("Training model...")
    model_config['num_embeddings'] = train_set.fields['text'].vocab.vectors.shape[0]
    model_config['embedding_dim'] = train_set.fields['text'].vocab.vectors.shape[1]

    model = Main(model_config, train_set.fields['text'].vocab)
    model.to(device)
    grad_params = [p for p in model.parameters() if p.requires_grad]
    weight = torch.FloatTensor(N_CLASSES).fill_(1)
    ce_loss = nn.CrossEntropyLoss(weight=weight).to(device)

    lr = LEARNING_RATE
    epoch = 1
    prev_dev_accuracy = 0
    optimizer = optim.SGD(grad_params, lr)
    print(THRESHOLD)
    best_epoch = 0
    while lr > THRESHOLD:
        # writer.add_scalar(
        #     'Learning rate', optimizer.param_groups[0]['lr'], epoch)
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * \
            WEIGHT_DECAY if epoch > 1 else optimizer.param_groups[0]['lr']
        train_it, val_it, test_it = data.get_batch_iterators(
            MINIBATCH_SIZE, train_set, val_set, test_set)
        train_loss = 0
        train_loss_epoch = 0
        batches = 0
        for batch in train_it:
            optimizer.zero_grad()
            output = model.forward(batch.text)
            loss = ce_loss(output, batch.label_a)
            loss.backward()

            train_loss += loss
            train_loss_epoch += loss
            optimizer.step()
            batches += 1
            if batches % 50 == 0:
                train_loss = 0
                # writer.add_scalar('Loss', train_loss, batches / 50)
        print("Training Loss at epoch: {} is: {}".format(epoch, train_loss))
        # writer.add_scalar('Loss (Epoch)', train_loss_epoch, epoch)
        n_correct = 0
        n_tested = 0
        # for batch in val_it:
        output = model.forward(batch.text)
        scores, predictions = torch.max(output, dim=1)

        n_correct += (batch.label_a == predictions).sum()
        n_tested += batch.label_a.shape[0]

        accuracy = n_correct.item()/n_tested
        # writer.add_scalar('Validation accuracy', accuracy, epoch)

        if accuracy < prev_dev_accuracy:
            optimizer.param_groups[0]['lr'] /= 5
            lr = optimizer.param_groups[0]['lr']
        else:
            prev_dev_accuracy = accuracy
            best_epoch = epoch

            torch.save(model, os.path.join(params.outputdir,
                                           params.model + "_epoch_" + str(epoch) + ".pt"))
        print("Validation accuracy at epoch: {} is: {}".format(
            epoch, accuracy))

        # Store model
        epoch += 1


if __name__ == "__main__":
    train()
    # validate_best()
