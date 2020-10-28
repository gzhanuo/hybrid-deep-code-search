import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F
import dataset.my_ast
import logging
logger = logging.getLogger(__name__)
import math, copy
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from model.attention import MultiHeadAttnRelative, MultiHeadAttn
from model.position_embedding import RelativePositionEmbedding, PositionalEncoding
from model.utils import gelu, subsequent_mask, clones, relative_mask

class BOWEncoder(nn.Module):
    '''
    https://medium.com/data-from-the-trenches/how-deep-does-your-sentence-embedding-model-need-to-be-cdffa191cb53
    https://www.kdnuggets.com/2019/10/beyond-word-embedding-document-embedding.html
    https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d#bbe8
    '''
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(BOWEncoder, self).__init__()
        self.emb_size=emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #self.word_weights = get_word_weights(vocab_size) 
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        
    def forward(self, input, input_len=None): 
        batch_size, seq_len =input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training) # [batch_size x seq_len x emb_size]
        
        # try to use a weighting scheme to summarize bag of word embeddings: 
        # for example, a smooth inverse frequency weighting algorithm: https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py
        # word_weights = self.word_weights(input) # [batch_size x seq_len x 1]
        # embeded = word_weights*embedded 
        
        # max pooling word vectors
        output_pool = F.max_pool1d(embedded.transpose(1,2), seq_len).squeeze(2) # [batch_size x emb_size]
        encoding = output_pool #torch.tanh(output_pool)        
        return encoding
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        for name, param in self.lstm.named_parameters(): # initialize the gate weights 
            # adopted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
            #if len(param.shape)>1:
            #    weight_init.orthogonal_(param.data) 
            #else:
            #    weight_init.normal_(param.data)                
            # adopted from fairseq
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, input_lens=None): 
        batch_size, seq_len=inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.4, self.training)
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            hids = F.dropout(hids, p=0.4, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
############commenting the following line significantly improves the performance, why? #####################################
        #h_n = h_n.transpose(1, 0).contiguous()# [batch_size x n_dirs x hid_sz]
        encoding = h_n.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        #pooled_encoding = F.max_pool1d(hids.transpose(1,2), seq_len).squeeze(2) # [batch_size x hid_size*2]
        #encoding = torch.tanh(pooled_encoding)

        return encoding #pooled_encoding

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)

'''
_____________________________________________hybrid-transformer modules___________________________________________________________________________
'''
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, code_embed, nl_embed, generator, num_heads):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_embed = code_embed
        self.nl_embed = nl_embed
        self.generator = generator
        self.num_heads = num_heads

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, nl, nl_mask = inputs

        encoder_code_mask = self.generate_code_mask(relative_par_ids, relative_bro_ids, semantic_ids)

        code_mask = (code == 0).unsqueeze(-2).unsqueeze(1)
        nl_embed = self.generate_nl_emb(nl)

        encoder_outputs = self.encode(code, relative_par_ids, relative_bro_ids, semantic_ids, encoder_code_mask)
        decoder_outputs, decoder_attn = self.decode(encoder_outputs, code_mask, nl_embed, nl_mask)
        return decoder_outputs, decoder_attn, encoder_outputs, nl_embed

    def encode(self, code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask):
        return self.encoder(self.code_embed(code), relative_par_ids, relative_bro_ids, semantic_ids, code_mask)

    def decode(self, memory, code_mask, nl_embed, nl_mask):
        return self.decoder(nl_embed, memory, code_mask, nl_mask)

    def generate_code_mask(self, relative_par_ids, relative_bro_ids, semantic_ids):
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0

        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], self.num_heads)
        return code_mask

    def generate_nl_emb(self, nl):
        return self.nl_embed(nl)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        output, attn = sublayer(self.norm(x))
        return x + self.dropout(output), attn


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x)))), None


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    def __init__(self, layer, N, relative_pos_emb, num_heads):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.relative_pos_emb = relative_pos_emb
        self.num_heads = num_heads

    def forward(self, code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask):
        relative_k_emb, relative_v_emb = self.relative_pos_emb([relative_par_ids, relative_bro_ids, semantic_ids])

        for layer in self.layers:
            code, attn = layer(code, code_mask, relative_k_emb, relative_v_emb)
        a=[]
        for i in range(len(code)):
            a.append(torch.mean(code[i],dim=0,keepdim=False))
        tup=(a[0],)
        for i in range(1,len(a)):
            tupsub=(a[i],)
            tup=tup+tupsub

        coderesult=torch.stack(tup, 0)
        # first_code_states  = code[:, 0, :]
        # return self.norm(first_code_states)
        return self.norm(coderesult)

        # code[i]=code[i].reshape(1,-1)
        # codeave=torch.mean(code[0],dim=0,keepdim=True)
        # coderoot=code[0][0].reshape(1,-1)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, code, mask, relative_k_emb, relative_v_emb):
        code, attn = self.sublayer[0](code, lambda x: self.self_attn(x, x, x, mask, relative_k_emb, relative_v_emb))
        output, _ = self.sublayer[1](code, self.feed_forward)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x), attn


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x, nl_attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x, code_attn = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        output, _ = self.sublayer[2](x, self.feed_forward)
        return output, code_attn


class PointerGenerator(nn.Module):
    def __init__(self, d_model, nl_vocab_size, semantic_begin, max_simple_name_len, dropout):
        super(PointerGenerator, self).__init__()

        self.d_model = d_model
        self.nl_vocab_size = nl_vocab_size
        self.p_vocab = nn.Sequential(
            nn.Linear(self.d_model, self.nl_vocab_size - max_simple_name_len),
            nn.Softmax(dim=-1)
        )
        self.p_gen = nn.Sequential(
            nn.Linear(3 * self.d_model, 1),
            nn.Sigmoid()
        )
        self.semantic_begin = semantic_begin
        self.soft_max = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_output, decoder_attn, memory, nl_embed, nl_convert, semantic_mask):
        """

        :param decoder_output: shape [batch_size, nl_len, d_model]
        :param decoder_attn: shape [batch_size, num_heads, nl_len, ast_len]
        :param memory: shape [batch_size, ast_len, d_model]
        :param nl_embed: shape [batch_size, nl_len, d_model]
        :param nl_convert: shape [batch_size, ast_len, max_simple_name_len]
        :param semantic_mask: shape [batch_size, ast_len]
        :return:
        """

        # shape [batch_size, nl_len, ast_len]
        decoder_attn = torch.sum(decoder_attn, dim=1)
        decoder_attn = decoder_attn.masked_fill(semantic_mask.unsqueeze(1) == 0, 0)

        p_vocab = self.p_vocab(decoder_output)  # shape [batch_size, nl_len, nl_vocab_size]
        context_vector = torch.matmul(decoder_attn, memory)  # shape [batch_size, nl_len, d_model]

        #  shape [batch_size, nl_len, 3 * d_model]
        total_state = torch.cat([context_vector, decoder_output, nl_embed], dim=-1)

        p_gen = self.p_gen(total_state)

        p_copy = 1 - p_gen

        # shape [batch_size, nl_len, max_simple_name_len]
        p_copy_ast = torch.matmul(decoder_attn, nl_convert)
        p_copy_ast = p_copy_ast.masked_fill(p_copy_ast == 0., 1e-9)
        p_copy_ast = self.soft_max(p_copy_ast)

        p = torch.cat([p_vocab * p_gen, p_copy_ast * p_copy], dim=-1)
        p = p.clamp(1e-9, 1.)
        p = torch.log(p)

        return p


def is_nan(inputs):
    return torch.sum(inputs != inputs) != 0


class Train(nn.Module):
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, semantic_convert_matrix, semantic_masks, nl, nl_mask = inputs
        decoder_outputs, decoder_attn, encoder_outputs, nl_embed = self.model.forward(
            (code, relative_par_ids, relative_bro_ids, semantic_ids, nl, nl_mask))
        out = self.model.generator(decoder_outputs, decoder_attn, encoder_outputs, nl_embed, semantic_convert_matrix, semantic_masks)
        return out


class GreedyEvaluate(nn.Module):
    def __init__(self, model,  max_nl_len, start_pos):
        super(GreedyEvaluate, self).__init__()
        self.model = model
        self.max_nl_len = max_nl_len
        self.start_pos = start_pos

    def forward(self, inputs):
        code, relative_par_ids, relative_bro_ids, semantic_ids, semantic_convert_matrix, semantic_masks, nl, nl_mask = inputs

        batch_size = code.size(0)

        encoder_code_mask = self.model.generate_code_mask(relative_par_ids, relative_bro_ids, semantic_ids)

        code_mask = (code == 0).unsqueeze(-2).unsqueeze(1)
        encoder_outputs = self.model.encode(code, relative_par_ids, relative_bro_ids, semantic_ids, encoder_code_mask)

        ys = torch.ones(batch_size, 1).fill_(self.start_pos).type_as(code.data)
        for i in range(self.max_nl_len - 1):
            nl_mask = Variable(subsequent_mask(ys.size(1)).type_as(code.data))
            nl_mask = (nl_mask == 0).unsqueeze(1)
            decoder_outputs, decoder_attn = self.model.decode(encoder_outputs,
                                                              code_mask,
                                                              self.model.generate_nl_emb(Variable(ys)),
                                                              nl_mask)

            prob = self.model.generator(decoder_outputs[:, -1].unsqueeze(1),
                                        decoder_attn[:, :, -1].unsqueeze(2),
                                        encoder_outputs,
                                        self.model.generate_nl_emb((Variable(ys)))[:, -1].unsqueeze(1),
                                        semantic_convert_matrix,
                                        semantic_masks)
            prob = prob.squeeze(1)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).type_as(code.data)], dim=1)
        return ys


def make_model(code_vocab, nl_vocab, N=2, d_model=300, d_ff=512, k=5, h=6,
               num_features=3, max_simple_name_len=30, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttn(d_model, h)
    attn_relative = MultiHeadAttnRelative(d_model, h)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn_relative), c(ff), dropout), N,
                RelativePositionEmbedding(d_model // h, k, h, num_features, dropout), h),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        Embeddings(d_model, code_vocab),
        nn.Sequential(Embeddings(d_model, nl_vocab), c(position)),
        PointerGenerator(d_model, nl_vocab, h // num_features * (num_features-1), max_simple_name_len, dropout),
        num_heads=h
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    make_model(50, 50)
 
 