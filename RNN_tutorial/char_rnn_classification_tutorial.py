"""
RNN으로 이름 분류하기, https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial
단어 분류를 위한 문자-단위 RNN implementation & training
NLP 모델링을 위한 데이터 전처리를 위한 라이브러리(torchtext)를 사용하지 않고
학습을 위해 데이터 전처리 과정 from scratch
주어진 이름이 어떤 언어로 이루어 졌는지 예측하기 위해,
18개 언어로 된 수천 개의 이름을 학습

download data : https://download.pytorch.org/tutorial/data.zip

Good explanation post for understanding RNN and LSTM, Korean
https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr
"""
# Data preparation, from scratch
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os


def findFiles(path): return glob.glob(path)


# check the name of all files for training
print(findFiles('data/names/*.txt'))

import unicodedata
import string

# all ASCII code char & count the number of them
all_letters = string.ascii_letters + " .,;"
n_letters = len(all_letters)

print(all_letters)


# 유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427
# unicode data normalize method : NFC, NFD, NFKD, NFKC
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != '\n'
        and c in all_letters
    )


# check Unicode to ASCII code
print(unicodeToAscii('Ślusàrski'))

# create 언어 종류 list & 언어별 학습 data
all_categories = []
category_lines = {}


# read file & split line by line >> return split file line list
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# read training data
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# check category type & contents(names)
print(all_categories)
print(category_lines['Italian'][:5])

'''
Name to Tensor : One-Hot vector
character - <1 * n_letters>
word - combine each char, <line_length * 1 * n_letters>
1 dim on the middle show batch 1, PyTorch assumes that everything is in batch
'''
import torch


# find the address of a character with all_letters, 'a': 0 / 'b': 1 / ...
def letterToiIndex(letter):
    return all_letters.find(letter)


# For verification, change one char to tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToiIndex(letter)] = 1
    return tensor


# one name(== one line) to One-Hot tensor array <line_length * 1 * n_letters>
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToiIndex(letter)] = 1
    return tensor


print(letterToTensor('J'))
print(lineToTensor('Jones').size())

'''
Create Network
'''
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# Example : running one step
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

# Example plus : For efficiency, make all tensors and cut them before to use
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)
