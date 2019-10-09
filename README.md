# DL4NLP-offenseval
Overleaf report: https://www.overleaf.com/8588136332ghqjwscnwwgw

## Training
`python train.py --model=transformer`

#### Parameters:
```
--model - str: transformer or lstm,  (default transformer)

--attention - BOOL: use attention for the LSTM (default False)

--epochs - int: training epochs (default 2)

--dropout - float: configure dropout in all layers (default 1E-3)

--weight-decay - float: configure weight decay (default 0.1)
```

## Evaluation
`python eval.py --checkpoint ./lstm_best_copy.pt`

#### Parameters:
```
--checkpoint: We have provided a checkpoint in the repository of the best LSTM model (+attention). It evaluates this trained model on the test set.
```

### Notes:
This runs the evaluations with 100 samples per example to approximate Bayesian Dropout.

## Infer
`python infer.py --checkpoint ./lstm_best_copy.pt --input-file in.txt --output-file out.txt`

#### Parameters:
```
--checkpoint We have provided a checkpoint in the repository of the best LSTM model, but you can also train one and replace it here.
--input-file This allows you to infer sentiment of tweets in a file (separated by commas)
--output-file Writes the predictions to this file 
```

