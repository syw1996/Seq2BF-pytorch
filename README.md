# Seq2BF: Based On OpenNMT-py

## Requirements

- PyTorch 0.4
- Python 3.6
- jieba

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

prepare forward train data:
```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data
```

prepare backward train data:
```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train-front.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val-front.txt -save_data data/front
```

Note the backward train must use the vocab created by forward train, don't use the vocab created by backward itself!!!

I will be working with some example data in `data/` folder(about one million pairs post and response).

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `tgt-train-front.txt`
* `src-val.txt`
* `tgt-val.txt`
* `tgt-val-front.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

train forward:
```bash
python train.py -data data/data -save_model forward_model/model
```

train backward:
```bash
python train.py -data data/data -save_model backward_model/model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 0` to use (say) GPU 0.

### Step 3: Translate

```bash
python translate.py -model forward_model/xxx.pt -backward_model backward_model/xxx.pt -replace_unk -verbose
```
Note the post is entered by terminal, each post need a keyword which is decided by yourself, you can also replace it with PMI which will automatically select a keyword based on post sentence.


