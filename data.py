import csv
import os
import shutil

import torch
from torchtext import data
from torchtext import datasets

data_folder = 'dataset'


def csv_to_dict(csv_path, delimiter=','):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        csv_dict = {}
        for line in reader:
            csv_dict[line[0]] = line[1]
    return csv_dict


def preprocess_data(data_folder=data_folder, train_size=0.8):
    text = data.Field(sequential=True, use_vocab=True,
                      init_token="<s>", eos_token="</s>",
                      include_lengths=True)

    label_a = data.LabelField(use_vocab=True)
    label_b = data.LabelField(use_vocab=True)
    label_c = data.LabelField(use_vocab=True)
    id_field = data.LabelField(use_vocab=False)

    # Check if transformed files exist. Otherwise create them
    transformed_path = os.path.join(data_folder, 'transformed')
    if not os.path.exists(transformed_path):
        os.makedirs(transformed_path)
        # Split train in train and dev
        with open(os.path.join(data_folder, 'olid-training-v1.0.tsv')) as training_set_tsv:
            with open(os.path.join(transformed_path, 'dev.csv'), 'w') as dev_set_csv:
                with open(os.path.join(transformed_path, 'train.csv'), 'w') as train_set_csv:
                    reader = csv.reader(training_set_tsv, delimiter='\t')
                    writer_train = csv.writer(train_set_csv)
                    writer_dev = csv.writer(dev_set_csv)
                    for idx, row in enumerate(reader):
                        if idx == 0:
                            # writer_train.writerow(row)
                            # writer_dev.writerow(row)
                            continue
                        if idx < 100:
                            writer_dev.writerow(row)
                            continue
                        writer_train.writerow(row)

        # Transform test set
        test_set_labels_a = csv_to_dict(
            os.path.join(data_folder, 'labels-levela.csv'))
        test_set_labels_b = csv_to_dict(
            os.path.join(data_folder, 'labels-levelb.csv'))
        test_set_labels_c = csv_to_dict(
            os.path.join(data_folder, 'labels-levelc.csv'))

        with open(os.path.join(transformed_path, 'test.csv'), 'w') as test_set_csv:
            writer = csv.writer(test_set_csv, delimiter=',')
            with open(os.path.join(data_folder, 'testset-levela.tsv')) as test_levela_tsv:
                reader = csv.reader(test_levela_tsv, delimiter='\t')
                for idx, row in enumerate(reader):
                    if idx == 0:
                        continue
                    label_a = 'NULL'
                    if row[0] in test_set_labels_a:
                        label_a = test_set_labels_a[row[0]]
                    row.append(label_a)

                    label_b = 'NULL'
                    if row[0] in test_set_labels_b:
                        label_b = test_set_labels_b[row[0]]
                    row.append(label_b)

                    label_c = 'NULL'
                    if row[0] in test_set_labels_c:
                        label_c = test_set_labels_c[row[0]]
                    row.append(label_c)

                    writer.writerow(row)

    # Load dataset
    train_set = data.TabularDataset(
        path=os.path.join(transformed_path, 'train.csv'), format='csv', fields=[('id', id_field), ('text', text), ('label_a', label_a), ('label_b', label_b), ('label_c', label_c)]
    )

    dev_set = data.TabularDataset(
        path=os.path.join(transformed_path, 'dev.csv'), format='csv', fields=[('id', id_field), ('text', text), ('label_a', label_a), ('label_b', label_b), ('label_c', label_c)]
    )

    test_set = data.TabularDataset(
        path=os.path.join(transformed_path, 'test.csv'),
        format='csv', fields=[('id', id_field), ('text', text), ('label_a', label_a), ('label_b', label_b), ('label_c', label_c)]
    )

    text.build_vocab(train_set, dev_set, test_set, vectors="glove.840B.300d")
    label_a.build_vocab(train_set, dev_set, test_set)
    label_b.build_vocab(train_set, dev_set, test_set)
    label_c.build_vocab(train_set, dev_set, test_set)

    return train_set, dev_set, test_set


def get_batch_iterators(batch_size, train_set, val_set, test_set):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_it, val_it, test_it = data.BucketIterator.splits(
        datasets=(train_set, val_set, test_set),
        batch_sizes=(
            batch_size, batch_size, batch_size),
        sort_key=lambda x: x.text,
        device=device,
        sort_within_batch=True,
        repeat=False,
    )
    return train_it, val_it, test_it


if __name__ == "__main__":
    train_set, val_set, test_set = preprocess_data()
    print(train_set.fields)
    print(train_set.fields['text'].vocab)
    print(train_set.fields['label_c'].vocab.stoi)
    train_it, val_it, test_it = get_batch_iterators(
        10, train_set, val_set, test_set)
    # for batch in train_it:
    #     print((batch.text))
    vocab = train_set.fields['text'].vocab
