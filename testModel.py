import os
import sys
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
from dataset.my_ast import read_pickle
from dataset.my_data_loader import DataLoaderX
from dataset.dataset import TreeDataSet
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
import dataset.my_ast
import torch
from utils import normalize
from data_loader import CodeSearchDataset, save_vecs
import models, configs
import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
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
try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME
except:
    IS_ON_NSML = False

os.chdir("C:/Users/Administrator/PycharmProjects/pytorch")


##### Compute Representation #####
def repr_code(args, ast2id, code2id, nl2id, id2nl):
    with torch.no_grad():
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        config = getattr(configs, 'config_' + args.model)()

        ##### Define model ######
        logger.info('Constructing Model..')
        logger.info(os.getcwd())
        model = getattr(models, args.model)(config, ast2id)  # initialize the model
        if args.reload_from > 0:
            ckpt_path = f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
        pool_size=100
        sim_measure='cos'
        #data_path = args.data_path + args.dataset + '/'
        '''
        use_set = eval(config['dataset_name'])(data_path, config['use_names'], config['name_len'],
                                  config['use_apis'], config['api_len'],
                                  config['use_tokens'], config['tokens_len'])
    
        data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=args.batch_size, 
                                      shuffle=False, drop_last=False, num_workers=1)
        '''
        valid_data_set = TreeDataSet(file_name=args.data_dir + '/train.json',
                                     ast_path=args.data_dir + '/tree/train/',
                                     ast2id=ast2id,
                                     nl2id=nl2id,
                                     max_ast_size=args.code_max_len,
                                     max_simple_name_size=args.max_simple_name_len,
                                     k=args.k,
                                     max_comment_size=args.comment_max_len,
                                     use_code=True,
                                     desc=config['valid_desc'],
                                     desclen=config['desc_len']
                                     )
        data_loader = DataLoaderX(dataset=valid_data_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,

                                  num_workers=2)
        accs, mrrs, maps, ndcgs = [], [], [], []
        code_reprs, desc_reprs = [], []
        n_processed = 0
        for batch in tqdm(data_loader):
            if len(batch) == 8:  # seq_tensor, rel_par, rel_bro, rel_semantic, descs, desc_len, bad_descs, bad_desc_len
                code_batch = [tensor.to(device).long() for tensor in batch[:4]]
                desc_batch = [tensor.to(device).long() for tensor in batch[4:6]]
            with torch.no_grad():

                code_repr = addCodeMaskToCalcuCodeRepr(model, *code_batch).data.cpu().numpy().astype(np.float32)
                desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(
                    np.float32)  # [poolsize x hid_size]
                if sim_measure == 'cos':
                    code_repr = normalize(code_repr)
                    desc_repr = normalize(desc_repr)
            code_reprs.append(code_repr)
            desc_reprs.append(desc_repr)
            n_processed += batch[0].size(0)
        code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)
        n_processed -= (n_processed % 100)
        for k in tqdm(range(0, n_processed - pool_size, pool_size)):
            code_pool, desc_pool = code_reprs[k:k + pool_size], desc_reprs[k:k + pool_size]
            sum = 0.0
            for i in range(min(10000, pool_size)):  # for i in range(pool_size):
                desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]

                if sim_measure == 'cos':
                    sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
                else:
                    sims = similarity(code_pool, desc_vec, sim_measure)  # [pool_size]
                if sims[i] > 0.4:
                    sum += 1;
                # negsims=np.negative(sims.T)
                # predict = np.argpartition(negsims, kth=n_results-1)#predict=np.argsort(negsims)#
                # predict = predict[:n_results]
                #
                # predict = [int(k) for k in predict]
                # real = [i]
                # for val in real:
                #     try:
                #         index = predict.index(val)
                #     except ValueError:
                #         index = -1
                #     if index != -1: sum = sum + 1

            accs.append(sum / float(pool_size))
            # accs.append(ACC(real,predict))
            # mrrs.append(MRR(real,predict))
            # maps.append(MAP(real,predict))
            # ndcgs.append(NDCG(real,predict))
        logger.info({'acc': np.mean(accs), 'err': 1 - np.mean(accs)})
        return {'acc': np.mean(accs), 'err': 1 - np.mean(accs)}

def addCodeMaskToCalcuCodeRepr(model,code, relative_par_ids, relative_bro_ids, semantic_ids):
    relative_par_mask = relative_par_ids == 0
    relative_bro_mask = relative_bro_ids == 0
    semantic_mask = semantic_ids == 0
    code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
    code_repr = model.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
    return code_repr

def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    #parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='dataset', help='dataset')
    parser.add_argument('--reload_from', type=int, default=-1, help='step to reload from')
    #parser.add_argument('--batch_size', type=int, default=10000,
                        #help='how many instances for encoding and normalization at each step')
    parser.add_argument('--chunk_size', type=int, default=2000000,
                        help='split code vector into chunks and store them individually. ' \
                             'Note: should be consistent with the same argument in the search.py')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    #parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    #parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')

    #parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    #parser.add_argument('--visual', default=False, help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=500, help='interval to validation')
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    i2code = read_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = read_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = read_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}
    repr_code(args, ast2id, code2id, nl2id, i2nl)


