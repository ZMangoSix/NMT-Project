import collections
import math
import torch
from torch import nn
from torch.nn import functional as F

import encoder_decoder as eder
from GRU import GRU


def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if 'weight' in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(eder.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # add model here
        self.model = GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)


    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X.t().type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.model(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state



class Seq2SeqDecoder(eder.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.model = GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)


    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
    

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(X.t().type(torch.int64))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.model(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]



class Seq2Seq(eder.EncoderDecoder):
    def __init__(self, encoder, decoder, padding_index, lr):
        super().__init__(encoder, decoder)
        self.criteria = nn.CrossEntropyLoss(ignore_index=padding_index)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)


    def loss(self, Y_hat, Y):
        return self.criteria(Y_hat, Y)


    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        batch = [a.to(device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights



def bleu(pred_seq, label_seq, k):
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score



if __name__ == '__main__':
    # Check encoder
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 9
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    X = torch.zeros((batch_size, num_steps))
    enc_outputs, enc_state = encoder(X)
    print(enc_outputs.shape, (num_steps, batch_size, num_hiddens))


    # Check decoder
    decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
    print(dec_outputs.shape, (batch_size, num_steps, vocab_size))
    print(state[1].shape, (num_layers, batch_size, num_hiddens))
    # d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
    # d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))


    model = Seq2Seq(encoder, decoder, 0, 0.005)
    print(model)

    # engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    # fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
