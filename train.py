import argparse
import os

from torch import nn
from torch import optim
from torch import Tensor
import torch

import data

from sklearn.metrics import confusion_matrix

from model import Main


LEARNING_RATE = 1e-3
LR_DECAY = 0.99
LR_DIVISION = 5
MINIBATCH_SIZE = 64
THRESHOLD = 10 ** -5
HIDDEN_LAYER_UNITS = 512
N_CLASSES = 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--datadir", type=str, default='dataset'),
parser.add_argument("--save_model", type=bool, default=True),
parser.add_argument("--outputdir", type=str,
                    default='savedir/', help="Output directory")
parser.add_argument("--model", type=str, default='transformer')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=20)

params, _ = parser.parse_known_args()

# writer = SummaryWriter(params.outputdir + 'runs/' + params.model)

model_config = {
    'num_embeddings': 300,
    'embedding_dim': 300,
    'input_dim': 1,
    'hidden_dim': HIDDEN_LAYER_UNITS,
    'n_classes': N_CLASSES,
    'lstm_dim': 128,
    'encoder': params.model,
    'dropout': params.dropout,
    'learning_rate': params.learning_rate,
    'weight_decay': params.weight_decay,
    'use_yang_attention': False
}


def adjust_learning_rate(optimizer, lr):
    """Sets the learning ratese to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # get batch iterator
    print("preprocess")
    train_set, val_set, test_set, _ = data.preprocess_data(
        data_folder=params.datadir)

    print(model_config)

    print("Training model...")
    model_config['num_embeddings'] = train_set.fields['text'].vocab.vectors.shape[0]
    model_config['embedding_dim'] = train_set.fields['text'].vocab.vectors.shape[1]

    model = Main(model_config, train_set.fields['text'].vocab)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print (pytorch_total_params)

    model.to(device)
    grad_params = [p for p in model.parameters() if p.requires_grad]
    # weight = torch.FloatTensor(N_CLASSES).fill_(1)
    weight = torch.FloatTensor([0.3, 0.7])
    ce_loss = nn.CrossEntropyLoss(weight=weight).to(device)

    lr = model_config['learning_rate']
    epoch = 1
    prev_dev_accuracy = 0
    optimizer = optim.Adam(grad_params, lr, weight_decay=model_config['weight_decay'])
    best_epoch = 0
    while epoch < params.epochs:
        # writer.add_scalar(
        #     'Learning rate', optimizer.param_groups[0]['lr'], epoch)
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * \
            LR_DECAY if epoch > 1 else optimizer.param_groups[0]['lr']
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
        true_positive = [0, 0]
        true_negative = [0, 0]
        false_positive = [0, 0]
        false_negative = [0, 0]
        true_labels = [0, 0]
        for batch in val_it:
            output = model.forward(batch.text)
            scores, predictions = torch.max(output, dim=1)

            for i in range(2):
                true_labels = (batch.label_a == i).nonzero()
                false_labels = (batch.label_a != i).nonzero()
                true_positive[i] += (predictions[true_labels]
                                     == i).sum().item()
                false_negative[i] += (predictions[true_labels]
                                      != i).sum().item()
                false_positive[i] += (predictions[false_labels]
                                      == i).sum().item()
                true_negative[i] += (predictions[false_labels]
                                     != i).sum().item()

            n_correct += (batch.label_a == predictions).sum()
            n_tested += batch.label_a.shape[0]

        # writer.add_scalar('Validation accuracy', accuracy, epoch)
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

        non_zero_devision = 1e-6
        for i in range(2):
            precision = (true_positive[i] /
                         (true_positive[i] + false_positive[i] + non_zero_devision))
            macro_precision += precision
            recall = (true_positive[i] /
                      (true_positive[i] + false_negative[i]+non_zero_devision))
            macro_recall += recall
            macro_f1 += 2 * (precision*recall) / \
                (precision+precision+ non_zero_devision)

        macro_precision /= 2
        macro_recall /= 2

        macro_f1 /= 2
        print(
            f'Precision: {macro_precision}\nRecall: {macro_recall}\nF1: {macro_f1}')
        accuracy = n_correct.item()/n_tested

        if macro_f1 <= prev_dev_accuracy:
            # optimizer.param_groups[0]['lr'] /= 2
            lr = optimizer.param_groups[0]['lr']
            print(f"lr: {lr} and threshold: {THRESHOLD}")
        else:
            prev_dev_accuracy = macro_f1
            best_epoch = epoch
            if (params.save_model):
                torch.save(model, os.path.join(params.outputdir,
                                               params.model + "_epoch_" + str(epoch) + ".pt"))

        print("Validation accuracy at epoch: {} is: {}, f1 {}".format(
            epoch, accuracy, macro_f1))

        # Store model
        epoch += 1
    validate(test_it, best_epoch,
             train_set.fields['text'].vocab, model_config)


def validate(test_it, best_epoch, vocab, model_config):
    # Load best model
    model = torch.load(os.path.join(params.outputdir,
                                    params.model + "_epoch_" + str(best_epoch) + ".pt"))

    model.eval()
    model.to(device)
    n_correct = 0
    n_tested = 0
    true_positive = [0, 0]
    true_negative = [0, 0]
    false_positive = [0, 0]
    false_negative = [0, 0]
    true_labels = [0, 0]
    for batch in test_it:
        output = model.forward(batch.text)
        scores, predictions = torch.max(output, dim=1)

        for i in range(2):
            true_labels = (batch.label_a == i).nonzero()
            false_labels = (batch.label_a != i).nonzero()
            true_positive[i] += (predictions[true_labels]
                                 == i).sum().item()
            false_negative[i] += (predictions[true_labels]
                                  != i).sum().item()
            false_positive[i] += (predictions[false_labels]
                                  == i).sum().item()
            true_negative[i] += (predictions[false_labels]
                                 != i).sum().item()

        n_correct += (batch.label_a == predictions).sum()
        n_tested += batch.label_a.shape[0]

    # writer.add_scalar('Validation accuracy', accuracy, epoch)
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for i in range(2):
        precision = (true_positive[i] /
                        (true_positive[i] + false_positive[i]))
        macro_precision += precision
        recall = (true_positive[i] /
                    (true_positive[i] + false_negative[i]))
        macro_recall += recall
        macro_f1 += 2 * (precision*recall) / \
            (precision+precision)

    macro_precision /= 2
    macro_recall /= 2

    macro_f1 /= 2
    print(
        f'Precision: {macro_precision}\nRecall: {macro_recall}\nF1: {macro_f1}')
    accuracy = n_correct.item()/n_tested

    print('Test accuracy', accuracy)
    torch.save(model, os.path.join(
        params.outputdir, params.model + "_best.pt"))
    # predict(model, test_it)

def get_prediction(model, x):
    model.eval()
    model.to(device)
    for batch in x:
        logits = model(batch)
        scores, predictions = torch.max(logits, dim=1)


def predict(model, x, num_samples=1000):
    prediction = get_prediction(model, x)
    model.train()
    model.to(device)
    y1 = []
    for batch in x:
        y1_batch = []
        for _ in range(num_samples):
            logits = model(batch)
            y1_batch.append(logits.numpy())
        y1.append(y1_batch)




if __name__ == "__main__":
    train()
    # validate_best()
