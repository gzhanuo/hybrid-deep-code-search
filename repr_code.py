import os
import sys
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm
from dataset.my_ast import read_pickle
from dataset.my_data_loader import DataLoaderX
from dataset.dataset import TreeDataSet, collate_fn
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
import dataset.my_ast
import torch
from utils import normalize
from data_loader import CodeSearchDataset, save_vecs
import models, configs    
os.chdir("C:/Users/Administrator/PycharmProjects/pytorch")
##### Compute Representation #####
def repr_code(args, ast2id, code2id, nl2id, id2nl):
 with torch.no_grad():
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config=getattr(configs, 'config_'+args.model)()

    ##### Define model ######
    logger.info('Constructing Model..')
    logger.info(os.getcwd())
    model = getattr(models, args.model)(config, ast2id)#initialize the model
    if args.reload_from>0:
        ckpt_path = f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
        model.load_state_dict(torch.load(ckpt_path, map_location=device))       
    model = model.to(device)   
    model.eval()

    data_path = args.data_path+args.dataset+'/'
    '''
    use_set = eval(config['dataset_name'])(data_path, config['use_names'], config['name_len'],
                              config['use_apis'], config['api_len'],
                              config['use_tokens'], config['tokens_len'])

    data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=args.batch_size, 
                                  shuffle=False, drop_last=False, num_workers=1)
    '''
    train_data_set = TreeDataSet(file_name=args.data_path + '/repr.json',

                                 ast_path=args.data_path + '/tree/repr/',
                                 ast2id=ast2id,
                                 nl2id=nl2id,
                                 max_ast_size=args.code_max_len,
                                 max_simple_name_size=args.max_simple_name_len,
                                 k=args.k,
                                 max_comment_size=args.comment_max_len,
                                 use_code=True,
                                 )

    data_loader = DataLoaderX(dataset=train_data_set,
                              batch_size=args.batch_size,
                              shuffle=True,

                              num_workers=2)




    chunk_id = 0
    vecs, n_processed = [], 0 
    for batch in tqdm(data_loader):
        torch.cuda.empty_cache()
        batch_gpu = [tensor.to(device).long() for tensor in batch]
        with torch.no_grad():
            reprs = model.code_encoding(*batch_gpu).data.cpu().numpy()
        reprs = reprs.astype(np.float32) # [batch x dim]
        if config['sim_measure']=='cos': # do normalization for fast cosine computation
            reprs = normalize(reprs)
        vecs.append(reprs)
        n_processed=n_processed+ batch[0].size(0)
        if n_processed>= args.chunk_size:
            output_path = f"{data_path}{config['use_codevecs'][:-3]}_part{chunk_id}.h5"
            save_vecs(np.vstack(vecs), output_path)
            chunk_id+=1
            vecs, n_processed = [], 0
    # save the last chunk (probably incomplete)
    output_path = f"{data_path}{config['use_codevecs'][:-3]}_part{chunk_id}.h5"
    save_vecs(np.vstack(vecs), output_path)
    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github', help='dataset')
    parser.add_argument('--reload_from', type=int, default=-1, help='step to reload from')
    parser.add_argument('--batch_size', type=int, default=10000, help='how many instances for encoding and normalization at each step')
    parser.add_argument('--chunk_size', type=int, default=2000000, help='split code vector into chunks and store them individually. '\
                        'Note: should be consistent with the same argument in the search.py')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    i2code = read_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = read_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = read_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}
    repr_code(args,ast2id,code2id,nl2id,i2nl)

        
   