import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from my_ast import *
from dataset.evaluation import load_json
from model.utils import pad_seq, subsequent_mask, make_std_mask
import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID, indexes2sent
from data_loader import load_dict, load_vecs
from utils import normalize, similarity, sent2indexes
import sys
sys.path.append("./")

class TreeDataSet(Dataset):
    def __init__(self,
                 file_name,
                 ast_path,
                 ast2id,
                 nl2id,
                 max_ast_size,
                 max_simple_name_size,
                 k,
                 max_comment_size,
                 use_code,
                 desc=None,
                 desclen=None):
        """
        :param file_name: 数据集名称
        :param ast_path: AST存放路径
        :param max_ast_size: 最大AST节点数
        :param k: 最大相对位置
        :param max_comment_size: 最大评论长度
        """
        super(TreeDataSet, self).__init__()
        print('loading data...')
        self.data_set = load_json(file_name)
        print('loading data finished...')

        self.max_ast_size = max_ast_size
        self.k = k
        self.max_comment_size = max_comment_size
        self.ast_path = ast_path
        self.ast2id = ast2id
        self.nl2id = nl2id
        self.max_simple_name_size = max_simple_name_size

        self.use_code = use_code
        self.desc = desc
        self.max_desc_len = desclen
        self.len = len(self.data_set)
        table_desc = tables.open_file('./data/' + desc)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]
        self.data_len = self.idx_descs.shape[0]
        if desc is not None:
            self.training=True

    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq=np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq=seq[:maxlen]
        return seq
    def __getitem__(self, index):
        data = self.data_set[index]
        ast_num = data['ast_num']
        nl = data['nl']
        '''
         nl:["<s>", "returns", "a", "0", "-", "based", "depth", "within", "the", "object", "graph", "of", "the", "current", "object", "being", "serialized", "</s>"]


        '''

        ast = read_pickle(self.ast_path + ast_num)
        seq, rel_par, rel_bro, rel_semantic, semantic_convert_matrix, semantic_mask = traverse_tree_to_generate_matrix(ast, self.max_ast_size, self.k, self.max_simple_name_size)

        seq_id = [self.ast2id[x] if x in self.ast2id else self.ast2id['<UNK>'] for x in seq]
        #nl_id = [self.nl2id[x] if x in self.nl2id else self.nl2id['<UNK>'] for x in nl]

        """to tensor"""
        seq_tensor = torch.LongTensor(seq_id)
        #nl_tensor = torch.LongTensor(pad_seq(nl_id, self.max_comment_size).long())


        if self.training:
            #good desc
            data_path = './data/vocab.desc.json'
            vocab_desc = load_dict(data_path)
            nl_len=len(nl)
            for i in range(nl_len):
                nl[i]=vocab_desc.get(nl[i], 3)

            #nl2index, nl_len = sent2indexes(nl, vocab_desc, 30)
            nl2long = np.array(nl).astype(np.long)
            good_desc_len = min(int(nl_len), self.max_desc_len)
            good_desc = nl2long
            good_desc = self.pad_seq(good_desc, self.max_desc_len)

            #bad_desc
            rand_offset = random.randint(0, self.len - 1)
            bad_seq=self.data_set[rand_offset]['nl']
            bad_len = len(bad_seq)
            for i in range(bad_len):
                bad_seq[i] = vocab_desc.get(bad_seq[i], 3)
            bad2long = np.array(bad_seq).astype(np.long)
            #bad_index, bad_len = sent2indexes(bad_seq, vocab_desc, 30)
            bad_desc_len = min(int(bad_len), self.max_desc_len)
            bad_desc = bad2long

            bad_desc = self.pad_seq(bad_desc , self.max_desc_len)

            return seq_tensor, rel_par, rel_bro, rel_semantic, good_desc, good_desc_len, bad_desc, bad_desc_len
        return seq_tensor, rel_par, rel_bro, rel_semantic

    def __len__(self):
        return self.len

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask


'''
def collate_fn(inputs):
    codes = []
    nls = []
    rel_pars = []
    rel_bros = []
    rel_semantics = []
    semantic_converts = []
    semantic_masks = []

    for i in range(len(inputs)):
        code, nl, rel_par, rel_bro, rel_semantic, semantic_convert, semantic_mask, desc, desclen = inputs[i]

        codes.append(code)
        nls.append(nl)
        rel_pars.append(rel_par)
        rel_bros.append(rel_bro)
        rel_semantics.append(rel_semantic)
        semantic_converts.append(semantic_convert)
        semantic_masks.append(semantic_mask)

    batch_code = torch.stack(codes, dim=0)
    batch_nl = torch.stack(nls, dim=0)

    batch_comments = batch_nl[:, :-1]
    batch_predicts = batch_nl[:, 1:]

    comment_mask = make_std_mask(batch_comments, 0)
    comment_mask = comment_mask.unsqueeze(1) == 0

    re_par_ids = torch.stack(rel_pars, dim=0)
    re_bro_ids = torch.stack(rel_bros, dim=0)
    rel_semantic_ids = torch.stack(rel_semantics, dim=0)
    semantic_converts = torch.stack(semantic_converts, dim=0)
    semantic_masks = torch.stack(semantic_masks, dim=0)

    return (batch_code, re_par_ids, re_bro_ids, rel_semantic_ids,desc,desclen), batch_predicts
'''
if __name__ == '__main__':
    '''tes=TreeDataSet(file_name=args.data_dir + '/train.json',

                                 ast_path=args.data_dir + '/tree/train/',
                                 ast2id=ast2id,
                                 nl2id=nl2id,
                                 max_ast_size=args.code_max_len,
                                 max_simple_name_size=args.max_simple_name_len,
                                 k=args.k,
                                 max_comment_size=args.comment_max_len,
                                 use_code=use_relative,
                                 desc=config['train_desc'],
                                 desclen=config['desc_len'])
    print(tes.test(1))
    '''
    table_desc = tables.open_file('./data/test.desc.h5')
    descs = table_desc.get_node('/phrases')[:].astype(np.long)
    idx_descs = table_desc.get_node('/indices')[:]
    len, pos = idx_descs[1]['length'], idx_descs[1]['pos']

    good_desc = descs[pos:pos + len]
    print(good_desc)