'''
Data load and pre-processing

This section be partly copied from pytorch tutorial.

- download data : https://download.pytorch.org/tutorial/data.zip
- pytorch tutorial : https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial
'''
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

import unicodedata
import string

import torch
import random

def findFiles(path): return glob.glob(path)\

all_letters = string.ascii_letters + " .,;"

def letterToiIndex(letter):
    return all_letters.find(letter)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != '\n'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


class NAME(object):
    n_letters = len(all_letters)
    
    def __init__(self, data_path, data_format):
        super(NAME, self)

        self.all_categories = []
        self.category_lines = {}

        # read training data
        for filename in findFiles(data_path + data_format):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)

    def getNLetters(self):
        return NAME.n_letters

    # For verification, change one char to tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, NAME.n_letters)
        tensor[0][letterToiIndex(letter)] = 1
        return tensor

    # one name(== one line) to One-Hot tensor array <line_length * 1 * n_letters>
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, NAME.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToiIndex(letter)] = 1
        return tensor

        # Change network output to likelihood category
    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)  # tensor max value & address
        category_i = top_i[0].item()  # tensor to int
        return self.all_categories[category_i], category_i


    # Select a random item from the list
    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]


    # Get random training example : a name and the language that makes it up
    def randomTrainingExample(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor