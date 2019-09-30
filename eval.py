import argparse

import torch

import data

from model import Main

import numpy as np

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate(test_it, model):
    # Load best model

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


def get_prediction(model, x):
    model.eval()
    model.to(device)
    predictions = np.array([])
    ground_truth = np.array([])
    for batch in x:
        logits = model(batch.text)
        scores, predictions_batch = torch.max(logits, dim=1)
        predictions_batch = predictions_batch.unsqueeze(
            1).cpu().detach().numpy()
        predictions = np.vstack([predictions, predictions_batch]) if predictions.size else predictions_batch
        ground_truth_batch = batch.label_a.unsqueeze(1).cpu().detach().numpy()
        ground_truth = np.vstack(
            [ground_truth, ground_truth_batch]) if ground_truth.size else ground_truth_batch
        

    return ground_truth, predictions

def predict(model, x, num_samples=1000):
    ground_truth, predictions = get_prediction(model, x)
    model.train()
    model.to(device)
    dropout_preds = np.array([])
    for batch in x:
        dropout_preds_batch = np.array([])
        for _ in range(num_samples):
            logits = torch.nn.functional.softmax(model(batch.text))[:, 1]
            dropout_preds_batch = np.vstack([dropout_preds_batch, logits.cpu().detach().numpy()]
                      ) if dropout_preds_batch.size else logits.cpu().detach().numpy()
        dropout_preds = np.vstack([dropout_preds, dropout_preds_batch.transpose((1, 0))]
                       ) if dropout_preds.size else dropout_preds_batch.transpose((1, 0))
    return ground_truth, predictions, dropout_preds

def get_confusion_data(ground_truth, prediction, dropout_pred):
    correct = ground_truth == prediction

    true_pos = (correct*prediction == 1)
    true_neg = (correct*prediction == 0)
    true_pos = dropout_pred[true_pos[:, 0], :]
    true_neg = dropout_pred[true_neg[:, 0], :]

    false_pos = ((1-correct)*prediction == 1)
    false_neg = ((1-correct)*prediction == 0)
    false_pos = dropout_pred[false_pos[:, 0], :]
    false_neg = dropout_pred[false_neg[:, 0], :]
    return true_pos, true_neg, false_pos, false_neg


def get_boxplots(data):
    n_models = data[0].shape[0]
    fig, axes = plt.subplots(2, 2)
    # going over TP, TN, FP, FN
    for j, index in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        concat_conf_data = []
        for i in range(n_models):
            conf_data = get_confusion_data(data[0][i], data[1][i], data[2][i])
            concat_conf_data.append(conf_data[j].mean(axis=1))
        axes[index].boxplot(concat_conf_data, positions=range(n_models))
    # fig.title('Confusion matrix')
    axes[(0, 0)].set_title('Truth: 1', fontsize=15)
    axes[(0, 1)].set_title('Truth: 0', fontsize=15)
    axes[(0, 0)].set_ylabel('Pred: 1', fontsize=15)
    axes[(1, 0)].set_ylabel('Pred: 0', fontsize=15)

    # axes[(1,1)].y
    plt.savefig('uncertainty')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Uncertainty estimation and evaluation')
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--datadir", type=str, default='dataset'),
    params, _ = parser.parse_known_args()

    print("preprocess")
    train_set, val_set, test_set = data.preprocess_data(
        data_folder=params.datadir)
    model = torch.load(params.checkpoint, map_location=device)
    train_it, val_it, test_it = data.get_batch_iterators(
            64, train_set, val_set, test_set)

    ground_truth, predictions, dropout_preds = predict(model,test_it, num_samples=100)
    np.savetxt('predictions.csv', predictions)
    np.savetxt('dropout_preds.csv', dropout_preds)
    np.savetxt('ground_truth.csv', ground_truth)
    ground_truth = ground_truth.reshape(
        1, ground_truth.shape[0], ground_truth.shape[1])
    predictions = predictions.reshape(1, predictions.shape[0], predictions.shape[1])
    dropout_preds = dropout_preds.reshape(1, dropout_preds.shape[0], dropout_preds.shape[1])
    
    data = (ground_truth, predictions, dropout_preds)

    # data = get_confusion_data(ground_truth, predictions, dropout_preds)
    get_boxplots(data)
    validate(test_it, model)
