import os
import torch
import xml.etree.ElementTree as ET

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
        tokenized_input = self.tokenizer(self.srcs, text_target=self.tgts,
                                         padding='max_length', truncation=True, max_length=self.max_length,
                                         return_tensors="pt")

        # Shift the decoder input by one index
        decoder_input_ids = tokenized_input['labels'][:, :-1].contiguous()
        tmp = torch.zeros([decoder_input_ids.shape[0], 1], dtype=torch.long)
        decoder_input_ids = torch.cat([tmp, decoder_input_ids], dim=1)

        return tokenized_input, decoder_input_ids

    def read_xml(self, file_tgt, file_src): 
        # Parse the XML file
<<<<<<< Updated upstream
        tree_tgt = ET.parse(file_tgt)
        root_tgt = tree_tgt.getroot()
        tree_src = ET.parse(file_src)
        root_src = tree_src.getroot()

        # Find all the <seg> elements and print their text content
        for seg in root_tgt.findall('.//seg'):
            target = seg.text
        for seg in root_src.findall('.//seg'):
            source = seg.text
        # print(root.tag)
        # for child in root:
        #     print(child.tag, child.attrib)
        self.srcs.append(''.join(source))
        self.tgts.append(''.join(target))
        print(f'Total number of samples: {len(self.srcs)}')
=======
        tree = ET.parse(filename)
        root = tree.getroot()
        # print(root.tag)
        # for child in root:
        #     print(child.tag, child.attrib)
        # Find all <seg> elements and print their 'id' attribute and text content
        for seg in root.findall('.//seg'):
            seg_id = seg.get('id')
            seg_text = seg.text.strip()  # .strip() removes leading/trailing whitespace
            print(f'ID: {seg_id}, Text: {seg_text}')
>>>>>>> Stashed changes


if __name__ == '__main__':
    data = DataSet()
    data.read_file('./data/fra.txt')
    data.tokenize()
    print(data.vocab_size)
