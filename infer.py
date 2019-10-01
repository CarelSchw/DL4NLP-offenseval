import argparse

import torch
import torchtext

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def infer_single(tweet, checkpoint):
    with open('temp_input', 'w') as temp:
        temp.write(tweet)
    infer(checkpoint, 'temp_input', 'temp_out')
    with open('temp_out') as temp:
        for line in temp:
            return line


def infer(checkpoint, input_file, output_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading checkpoint...")
    print (checkpoint)
    model = torch.load(checkpoint, map_location=device)

    print("Building vocabulary...")
    dataset = build_dataset(input_file)

    print("Inferring inference...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    iterator, _ = torchtext.data.BucketIterator.splits(
        datasets=(dataset, dataset),
        batch_sizes=(64, 64),
        # on what attribute the text should be sorted
        device=device,
        sort_within_batch=False,
        repeat=False,
        shuffle=False,
    )

    vocab = dataset.fields['text'].vocab
    model.embedding = torch.nn.Embedding(
        vocab.vectors.shape[0], 300)
    model.embedding.weight.data.copy_(vocab.vectors)
    model.eval()
    model.to(device)
    predictions = []
    i = 0
    for b in iterator:
        output = model.forward(b.text)
        y = torch.argmax(output, dim=1)
        y.detach().cpu().numpy()
        for p in y.tolist():
            if p == 1:
                predictions.append("{}-O\n".format(i))
            if p == 0:
                predictions.append("{}-N\n".format(i))
            i+=1

    with open(output_file, 'w') as output_file:
        output_file.writelines(predictions)

def evaluate_sentences(input_file, predictions):
    sentences = []
    


def build_dataset(input_file_loc):
    print("Building vocabulary...")
    examples = []
    with open(input_file_loc) as input_file:
        for line in input_file:

            examples.append((line[:-2]))

    TEXT = torchtext.data.Field(sequential=True, use_vocab=True,
                                init_token="<s>", eos_token="</s>", include_lengths=True, tokenize="spacy")
    samples = [torchtext.data.Example.fromlist(
        [x], [("text", TEXT)]) for x in examples]

    dataset = torchtext.data.Dataset(
        samples, {"text": TEXT})
    TEXT.build_vocab(dataset,
                     vectors="glove.840B.300d")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLI inferring')
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    params, _ = parser.parse_known_args()
    infer(params.checkpoint, params.input_file, params.output_file)