import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

import torch

from utils import normalize, similarity, sent2indexes
from data_loader import load_dict, load_vecs
import models, configs
import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import torch.nn.functional as F
import argparse
random.seed(42)
from tqdm import tqdm
import dataset.my_ast
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package
from dataset.my_ast import read_pickle
import torch
import dataset.my_ast
import models, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *
from dataset.my_data_loader import DataLoaderX
from dataset.dataset import TreeDataSet
from model.utils import gelu, subsequent_mask, clones, relative_mask
os.chdir("C:/Users/Administrator/PycharmProjects/pytorch")
codevecs, codebase = [], []

##### Data Set #####   
def load_codebase(code_path, chunk_size=2000000):
    """load codebase
      codefile: h5 file that stores raw code
    """
    logger.info(f'Loading codebase (chunk size={chunk_size})..')
    codebase= []
    codes = codecs.open(code_path, encoding='latin-1').readlines() # use codecs to read in case of encoding problem
    for i in range(0, len(codes), chunk_size):
        codebase.append(codes[i: i+chunk_size]) 
    '''
    import subprocess
    n_lines = int(subprocess.check_output(["wc", "-l", code_path], universal_newlines=True).split()[0])
    for i in range(1, n_lines+1, chunk_size):
        codecs = subprocess.check_output(["sed",'-n',f'{i},{i+chunk_size}p', code_path]).split()
        codebase.append(codecs)
   '''
    return codebase

### Results Data ###
def load_codevecs(vec_path, chunk_size=2000000):
    logger.debug(f'Loading code vectors (chunk size={chunk_size})..')       
    """read vectors (2D numpy array) from a hdf5 file"""
    codevecs=[]
    chunk_id = 0
    chunk_path = f"{vec_path[:-3]}_part{chunk_id}.h5"
    while os.path.exists(chunk_path):
        reprs = load_vecs(chunk_path)
        codevecs.append(reprs)
        chunk_id+=1
        chunk_path = f"{vec_path[:-3]}_part{chunk_id}.h5"
    return codevecs

def search(config, model, vocab, query, n_results=10):
    model.eval()
    device = next(model.parameters()).device
    desc, desc_len =sent2indexes(query, vocab_desc, config['desc_len'])#convert query into word indices
    desc = torch.from_numpy(desc).unsqueeze(0).to(device)
    desc_len = torch.from_numpy(desc_len).clamp(max=config['desc_len']).to(device)
    # data_path = './data/vocab.desc.json'
    # vocab_desc = load_dict(data_path)
    # nl_len = len(nl)
    # for i in range(nl_len):
    #     nl[i] = vocab_desc.get(nl[i], 3)
    # # print(nl)
    # # print("\n")
    # # nl2index, nl_len = sent2indexes(nl, vocab_desc, 30)
    # nl2long = np.array(nl).astype(np.long)
    # good_desc_len = min(int(nl_len), self.max_desc_len)
    # good_desc = nl2long
    # good_desc = self.pad_seq(good_desc, self.max_desc_len)
    with torch.no_grad():
        desc_repr = model.desc_encoding(desc, desc_len).data.cpu().numpy().astype(np.float32) # [1 x dim]
    if config['sim_measure']=='cos': # normalizing vector for fast cosine computation
        desc_repr = normalize(desc_repr) # [1 x dim]
    results =[]
    threads = []
    for i, codevecs_chunk in enumerate(codevecs):
        t = threading.Thread(target=search_thread, args = (results, desc_repr, codevecs_chunk, i, n_results, config['sim_measure']))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:#wait until all sub-threads have completed
        t.join()
    return results

def search_thread(results, desc_repr, codevecs, i, n_results, sim_measure):        
#1. compute code similarities
    if sim_measure=='cos':
        chunk_sims = np.dot(codevecs, desc_repr.T)[:,0] # [pool_size]
    else:
        chunk_sims = similarity(codevecs, desc_repr, sim_measure) # [pool_size]
    
#2. select the top K results
    negsims = np.negative(chunk_sims)
    maxinds = np.argpartition(negsims, kth=n_results-1)
    maxinds = maxinds[:n_results]  
    chunk_codes = [codebase[i][k] for k in maxinds]
    chunk_sims = chunk_sims[maxinds]
    results.extend(zip(chunk_codes, chunk_sims))
    
def postproc(codes_sims):
    codes_, sims_ = zip(*codes_sims)
    codes = [code for code in codes_]
    sims = [sim for sim in sims_]
    final_codes = []
    final_sims = []
    n = len(codes_sims)        
    for i in range(n):
        is_dup=False
        for j in range(i):
            if codes[i][:80]==codes[j][:80] and abs(sims[i]-sims[j])<0.01:
                is_dup=True
        if not is_dup:
            final_codes.append(codes[i])
            final_sims.append(sims[i])
    return zip(final_codes,final_sims)
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--visual', default=False, help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=100, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # Model Hyperparameters for automl tuning
    # parser.add_argument('--emb_size', type=int, default=-1, help = 'embedding dim')
    parser.add_argument('--n_hidden', type=int, default=-1,
                        help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default=-1)
    parser.add_argument('--margin', type=float, default=-1)
    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')

    parser.add_argument('--learning_rate', type=float, help='learning rate')
    # parser.add_argument('--adam_epsilon', type=float)
    # parser.add_argument("--weight_decay", type=float, help="Weight deay if we apply some.")
    # parser.add_argument('--warmup_steps', type=int)

    # reserved args for automl pbt
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)

    ##############################################################################################################################
    # parser = argparse.ArgumentParser(description='tree transformer')
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=7)
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-num_layers', type=int, default=2, help='layer num')
    parser.add_argument('-model_dim', type=int, default=384)
    parser.add_argument('-num_heads', type=int, default=6)
    parser.add_argument('-ffn_dim', type=int, default=1536)

    parser.add_argument('-data_dir', default='./data')
    parser.add_argument('-dataset', default='./dataset')
    parser.add_argument('-datasave', default='/codevec')
    parser.add_argument('-code_max_len', type=int, default=100, help='max length of code')
    parser.add_argument('-comment_max_len', type=int, default=30, help='comment max length')
    parser.add_argument('-relative_pos', type=bool, default=True, help='use relative position')
    parser.add_argument('-k', type=int, default=5, help='relative window size')
    parser.add_argument('-max_simple_name_len', type=int, default=30, help='max simple name length')

    parser.add_argument('-dropout', type=float, default=0.5)

    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')

    parser.add_argument('-load_epoch', type=str, default='0')

    parser.add_argument('-log_dir', default='train_log/')

    parser.add_argument('-g', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=2000000,
                        help='split code vector into chunks and store them individually. ' \
                             'Note: should be consistent with the same argument in the search.py')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    i2code = read_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = read_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = read_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_'+args.model)()
    
    ##### Define model ######
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config, ast2id)#initialize the model
    ckpt=f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    data_path = args.data_path+args.datasave+'/'
    
    vocab_desc = load_dict(data_path+config['vocab_desc'])
    codebase = load_codebase(data_path+config['use_codebase'], args.chunk_size)
    codevecs = load_codevecs(data_path+config['use_codevecs'], args.chunk_size)
    assert len(codebase)==len(codevecs), \
         "inconsistent number of chunks, check whether the specified files for codebase and code vectors are correct!"    
    
    while True:
        try:
            query = input('Input Query: ')
            n_results = int(input('How many results? '))
        except Exception:
            print("Exception while parsing your input:")
            traceback.print_exc()
            break
        query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
        results = search(config, model, vocab_desc, query, n_results)
        results = sorted(results, reverse=True, key=lambda x:x[1])
        results = postproc(results)
        results = list(results)[:n_results]
        results = '\n\n'.join(map(str,results)) #combine the result into a returning string
        print(results)

