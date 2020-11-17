import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
import dataset.my_ast
import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import *
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

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
class JointEmbeder(nn.Module):
    """
    https://arxiv.org/pdf/1508.01585.pdf
    https://arxiv.org/pdf/1908.10084.pdf
    """
    def __init__(self, config, ast2id):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.ast2id=ast2id
        self.margin = config['margin']
        c = copy.deepcopy
        attn = MultiHeadAttn(config['d_model'], config['h'])
        attn_relative = MultiHeadAttnRelative(config['d_model'], config['h'])
        ff = PositionWiseFeedForward(config['d_model'], config['d_ff'], config['dropout'])
        position = PositionalEncoding(config['d_model'], config['dropout'])
        self.code_encoder=Encoder(EncoderLayer(config['d_model'], c(attn_relative), c(ff), config['dropout']), config['N'],
                RelativePositionEmbedding(config['d_model'] // config['h'], config['k'], config['h'], config['num_features'], config['dropout']), config['h'])
        ################self.name_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        ################self.api_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        ################self.tok_encoder=BOWEncoder(config['n_words'],config['emb_size'],config['n_hidden'])

        self.code_embed=Embeddings(config['d_model'],len(self.ast2id))
        self.desc_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        #self.fuse1=nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden'])
        #self.fuse2 = nn.Sequential(
        #    nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden']),
        #    nn.BatchNorm1d(config['n_hidden'], eps=1e-05, momentum=0.1),
        #    nn.ReLU(),
        #    nn.Linear(config['n_hidden'], config['n_hidden']),
        #)
        ##############self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        ##############self.w_api = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        ##############self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_desc = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.fuse3 = nn.Linear(config['n_hidden'], config['n_hidden'])
        
        ##############self.init_weights()
        
    def init_weights(self):# Initialize Linear Weight 
        for m in [self.w_name, self.w_api, self.w_tok, self.fuse3]:        
            m.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.) 
            
    def code_encoding(self, code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask):
        #def code_encoding(self, name, name_len, api, api_len, tokens, tok_len):
        ##############name_repr=self.name_encoder(name, name_len)
        ##############api_repr=self.api_encoder(api, api_len)
        ################tok_repr=self.tok_encoder(tokens, tok_len)
        #code_repr= self.fuse2(torch.cat((name_repr, api_repr, tok_repr),1))
        ################code_repr = self.fuse3(torch.tanh(self.w_name(name_repr)+self.w_api(api_repr)+self.w_tok(tok_repr)))
        code_repr=self.code_encoder(self.code_embed(code), relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
        return code_repr
        
    def desc_encoding(self, desc, desc_len):
        desc_repr=self.desc_encoder(desc, desc_len)
        desc_repr=self.w_desc(desc_repr)
        return desc_repr
    
    def similarity(self, code_vec, desc_vec):
        """
        https://arxiv.org/pdf/1508.01585.pdf 
        """
        assert self.conf['sim_measure'] in ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"
        if self.conf['sim_measure']=='cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.conf['sim_measure']=='poly':
            return (0.5*torch.matmul(code_vec, desc_vec.t()).diag()+1)**2
        elif self.conf['sim_measure']=='sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag()+1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2) # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure']=='euc': return euc_sim                
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag()+1)
            if self.conf['sim_measure']=='gesd': 
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure']=='aesd':
                return 0.5*(euc_sim+sigmoid_sim)
    
    def forward(self, code, relative_par_ids, relative_bro_ids, semantic_ids, negcode, negrelative_par_ids, negrelative_bro_ids, negsemantic_ids,negcode2, negrelative_par_ids2, negrelative_bro_ids2, negsemantic_ids2,negcode3, negrelative_par_ids3, negrelative_bro_ids3, negsemantic_ids3, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        batch_size=code.size(0)
        rand=random.randint(0,1)
        negcode_repr = []
        neg_sims = []
        #0:neg   1:pos
        ###############code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        # --------------------------poscode and posdesc-------------------------------
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0
        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
        code_repr = self.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        # desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)

        # ----------------------------negcode------------------------------
        #1
        negrelative_par_mask = negrelative_par_ids == 0
        negrelative_bro_mask = negrelative_bro_ids == 0
        negsemantic_mask = negsemantic_ids == 0
        negcode_mask = relative_mask([negrelative_par_mask, negrelative_bro_mask, negsemantic_mask], 6)
        negcode_repr = self.code_encoding(negcode, negrelative_par_ids, negrelative_bro_ids, negsemantic_ids,
                                          negcode_mask)
        neg_sim1 = self.similarity(negcode_repr, desc_anchor_repr)
        #2
        negrelative_par_mask2 = negrelative_par_ids2 == 0
        negrelative_bro_mask2 = negrelative_bro_ids2 == 0
        negsemantic_mask2 = negsemantic_ids2 == 0
        negcode_mask2 = relative_mask([negrelative_par_mask2, negrelative_bro_mask2, negsemantic_mask2], 6)
        negcode_repr2 = self.code_encoding(negcode2, negrelative_par_ids2, negrelative_bro_ids2, negsemantic_ids2,
                                          negcode_mask2)
        neg_sim2 = self.similarity(negcode_repr2, desc_anchor_repr)
        #3
        negrelative_par_mask3 = negrelative_par_ids3 == 0
        negrelative_bro_mask3 = negrelative_bro_ids3 == 0
        negsemantic_mask3 = negsemantic_ids3 == 0
        negcode_mask3 = relative_mask([negrelative_par_mask3, negrelative_bro_mask3, negsemantic_mask3], 6)
        negcode_repr3 = self.code_encoding(negcode3, negrelative_par_ids3, negrelative_bro_ids3, negsemantic_ids3,
                                          negcode_mask3)
        neg_sim3 = self.similarity(negcode_repr3, desc_anchor_repr)

        neg_sim = torch.stack((neg_sim1, neg_sim2, neg_sim3), 0)
        neg_simmean = neg_sim.mean(0)

        # print("pos:")
        # print(anchor_sim)
        # print("neg:")
        # print(neg_sim)
        loss = (self.margin+neg_simmean-anchor_sim).clamp(min=1e-6).mean()

        #test if no neg what happened
        # loss = (self.margin- anchor_sim).clamp(min=1e-6).mean()
            #print("0")
            #print("neg:")
            #print(neg_sim)
        # else:
        #     loss=(self.margin-anchor_sim).clamp(min=1e-6).mean()
        #     #print("1")
        #     #print("pos:")
        #     #print(anchor_sim)
        return loss

    def forward2(self, code, relative_par_ids, relative_bro_ids, semantic_ids, desc_anchor, desc_anchor_len, desc_neg,
                desc_neg_len):
        batch_size = code.size(0)
        rand = random.randint(0, 1)
        # 0:neg   1:pos
        ###############code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0
        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
        code_repr = self.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        # desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        # neg_sim = self.similarity(code_repr, desc_neg_repr)  # [batch_sz x 1]

        return anchor_sim
    def getcodevec(self, code, relative_par_ids, relative_bro_ids, semantic_ids, negcode, negrelative_par_ids, negrelative_bro_ids, negsemantic_ids,negcode2, negrelative_par_ids2, negrelative_bro_ids2, negsemantic_ids2,negcode3, negrelative_par_ids3, negrelative_bro_ids3, negsemantic_ids3, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        batch_size = code.size(0)
        rand = random.randint(0, 1)
        # 0:neg   1:pos
        ###############code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0
        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
        code_repr = self.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)

        return code_repr
    def getdescvec(self, code, relative_par_ids, relative_bro_ids, semantic_ids, negcode, negrelative_par_ids, negrelative_bro_ids, negsemantic_ids,negcode2, negrelative_par_ids2, negrelative_bro_ids2, negsemantic_ids2,negcode3, negrelative_par_ids3, negrelative_bro_ids3, negsemantic_ids3, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        batch_size = code.size(0)
        rand = random.randint(0, 1)
        # 0:neg   1:pos
        ###############code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        relative_par_mask = relative_par_ids == 0
        relative_bro_mask = relative_bro_ids == 0
        semantic_mask = semantic_ids == 0
        code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
        code_repr = self.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)

        return desc_anchor_repr