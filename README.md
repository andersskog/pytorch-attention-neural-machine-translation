# Pytorch Attention Neural Machine Translation

Paper implementation of [Neural Machine Translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf)

## Requirements

- Python 3.6
- PyTorch

## Installation

```
git clone https://github.com/andersskog/pytorch-attention-neural-machine-translation.git
cd pytorch-attention-neural-machine-translation
pip3 install -r requirements.txt
```
Note: for MacOS, see http://pytorch.org/ for other OS.

## Usage

```
python3 trainer.py language_in language_out dataset_path
```

### Example

```
python3 trainer.py en vi data
```
This will work when english and vietnamese datasets are available and datasets are in data folder.

