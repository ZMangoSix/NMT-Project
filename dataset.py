import os
import torch
from transformers import MarianTokenizer


# Currently used for Tatoeba Project only
class DataSet():
    def __init__(self, max_length=16):
        self.srcs = list()
        self.tgts = list()
        self.tokens = list()
        self.max_length = max_length
        # Maybe define a tokenizer here?
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

        self.vocab_size = self.tokenizer.vocab_size

    
    def read_file(self, filename):

        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '

        with open(filename, encoding='utf-8') as f:
            for line in f:
                tmp = line.split('\t')
                source = tmp[0].replace('\u202f', ' ').replace('\xa0', ' ')
                target = tmp[1].replace('\u202f', ' ').replace('\xa0', ' ')
                source = [' ' + char if i > 0 and no_space(char, source[i - 1]) else char
                          for i, char in enumerate(source.lower())]
                target = [' ' + char if i > 0 and no_space(char, target[i - 1]) else char
                          for i, char in enumerate(target.lower())]
                self.srcs.append(''.join(source))
                self.tgts.append(''.join(target))
        print(f'Total number of samples: {len(self.srcs)}')
        # print(self.srcs[:10])
        # print(self.tgts[:10])


    def tokenize(self):
        tokenized_input = self.tokenizer(self.srcs[:3], text_target=self.tgts[:3],
                                         padding='max_length', truncation=True, max_length=self.max_length,
                                         return_tensors="pt")

        # Shift the decoder input by one index
        decoder_input_ids = tokenized_input['labels'][:, :-1].contiguous()
        tmp = torch.zeros([decoder_input_ids.shape[0], 1], dtype=torch.long)
        decoder_input_ids = torch.cat([tmp, decoder_input_ids], dim=1)

        return tokenized_input, decoder_input_ids




if __name__ == '__main__':
    data = DataSet()
    data.read_file('./data/fra.txt')
    data.tokenize()
    print(data.vocab_size)
