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
def bind_nsml(model, **kwargs):
    if type(model) == torch.nn.DataParallel: model = model.module
    def infer(raw_data, **kwargs):
        pass
    def load(path, *args):
        weights = torch.load(path)
        model.load_state_dict(weights)
        logger.info(f'Load checkpoints...!{path}')
    def save(path, *args):
        torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))
        logger.info(f'Save checkpoints...!{path}')
    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)

    
def train(args, ast2id, code2id, nl2id, id2nl):
    nl_vocab_size = len(nl2id)
    use_relative = True
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logs.txt")
                                      # create file handler which logs even debug messages
    logger.addHandler(fh)# add the handlers to the logger
    timestamp = datetime.now().strftime('%Y%m%d%H%M')

    tb_writer = SummaryWriter(f"./output/{args.model}/{args.dataset}/logs/{timestamp}") if args.visual else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu") 

    config=getattr(configs, 'config_'+args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)
    
    ###############################################################################
    # Load data
    ###############################################################################
    data_path = DATASET_PATH+"/train/" if IS_ON_NSML else args.data_path+args.dataset+'/'
    '''
    train_set = eval(config['dataset_name'])(data_path, config['train_name'], config['name_len'],
                                  config['train_api'], config['api_len'],
                                  config['train_tokens'], config['tokens_len'],
                                  config['train_desc'], config['desc_len'])
    valid_set = eval(config['dataset_name'])(data_path,
                                  config['valid_name'], config['name_len'],
                                  config['valid_api'], config['api_len'],
                                  config['valid_tokens'], config['tokens_len'],
                                  config['valid_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], 
                                       shuffle=True, drop_last=True, num_workers=1)
    '''
    train_data_set = TreeDataSet(file_name=args.data_dir + '/train.json',

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

    data_loader = DataLoaderX(dataset=train_data_set,
                               batch_size=args.batch_size,
                               shuffle=True,

                               num_workers=2)

    ###############################################################################
    # Define Model
    ###############################################################################
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config, ast2id)#initialize the model
    
    def save_model(model, ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))
        
    if args.reload_from>0:
        ckpt = f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
        load_model(model, ckpt, device)    
        
    if IS_ON_NSML:
        bind_nsml(model)
    model.to(device)    
    
    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])        
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config['warmup_steps'], 
            num_training_steps=len(data_loader)*config['nb_epoch']) # do not foget to modify the number when dataset is changed
    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])
    
    ###############################################################################
    # Training Process
    ###############################################################################    
    n_iters = len(data_loader)
    itr_global = args.reload_from+1
    code_reprs, desc_reprs = [], []
    for epoch in range(int(args.reload_from/n_iters)+1, config['nb_epoch']+1): 
        itr_start_time = time.time()
        losses=[]
        n_processed = 0
        for batch in data_loader:
            
            model.train()
            batch_gpu = [tensor.to(device).long() for tensor in batch]
            loss = model(*batch_gpu)

            # code_repr=normalize(code_repr.data.cpu().numpy().astype(np.float32))
            # desc_repr = normalize(desc_repr.data.cpu().numpy().astype(np.float32))
            # code_reprs.append(code_repr)
            # desc_reprs.append(desc_repr)

            #n_processed += batch[0].size(0)
            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            losses.append(loss.item())
            
            if itr_global % args.log_every ==0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f'%
                        (epoch, config['nb_epoch'], itr_global%n_iters, n_iters, elapsed, np.mean(losses)))
                if tb_writer is not None:
                    tb_writer.add_scalar('loss', np.mean(losses), itr_global)
                if IS_ON_NSML:
                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update({'loss':np.mean(losses)})
                    nsml.report(**summary)
                    
                losses=[] 
                itr_start_time = time.time() 
            itr_global = itr_global + 1

            if itr_global % args.valid_every == 0:
                logger.info("validating..")
                with torch.no_grad():
                    valid_result = validate(model, config['pool_size'], config['top_k'], config['sim_measure'])
                logger.info(valid_result)
                if tb_writer is not None:
                    for key, value in valid_result.items():
                        tb_writer.add_scalar(key, value, itr_global)
                if IS_ON_NSML:
                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update(valid_result)
                    nsml.report(**summary)
                code_reprs, desc_reprs = [], []
                    
            if itr_global % args.save_every == 0:
                ckpt_path = f'./output/{args.model}/{args.dataset}/models/step{itr_global}.h5'
                save_model(model, ckpt_path)
                if IS_ON_NSML:
                    nsml.save(checkpoint=f'model_step{itr_global}')

##### Evaluation #####
def validate(model, pool_size, K, sim_measure):
    """
    simple validation in a code pool. 
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """
    config = getattr(configs, 'config_' + args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)
    def ACC(real,predict):
        sum=0.0
        for val in real:
            try: index=predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+1  
        return sum/float(len(real))
    def MAP(real,predict):
        sum=0.0
        for id, val in enumerate(real):
            try: index=predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+(id+1)/float(index+1)
        return sum/float(len(real))
    def MRR(real, predict):
        sum=0.0
        for val in real:
            try: index = predict.index(val)
            except ValueError: index=-1
            if index!=-1: sum=sum+1.0/float(index+1)
        return sum/float(len(real))
    def NDCG(real, predict):
        dcg=0.0
        idcg=IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i+1
                dcg +=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
        return dcg/float(idcg)
    def IDCG(n):
        idcg=0
        itemRelevance=1
        for i in range(n): idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
        return idcg

    model.eval()
    device = next(model.parameters()).device
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
    accs, mrrs, maps, ndcgs=[],[],[],[]
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in tqdm(data_loader):
        if len(batch) == 8: # seq_tensor, rel_par, rel_bro, rel_semantic, descs, desc_len, bad_descs, bad_desc_len
            code_batch = [tensor.to(device).long() for tensor in batch[:4]]
            desc_batch = [tensor.to(device).long() for tensor in batch[4:6]]
        with torch.no_grad():

            code_repr=addCodeMaskToCalcuCodeRepr(model,*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr=model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32) # [poolsize x hid_size]
            if sim_measure=='cos':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)
    n_processed-=(n_processed%100)
    for k in tqdm(range(0, n_processed-pool_size, pool_size)):
        code_pool, desc_pool = code_reprs[k:k+pool_size], desc_reprs[k:k+pool_size]
        sum=0.0
        for i in range(min(10000, pool_size)): # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0) # [1 x dim]
            n_results = K    
            if sim_measure=='cos':
                sims = np.dot(code_pool, desc_vec.T)[:,0] # [pool_size]
            else:
                sims = similarity(code_pool, desc_vec, sim_measure) # [pool_size]
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

        accs.append(sum/float(pool_size))
            # accs.append(ACC(real,predict))
            # mrrs.append(MRR(real,predict))
            # maps.append(MAP(real,predict))
            # ndcgs.append(NDCG(real,predict))
    return {'acc':np.mean(accs), 'err': 1-np.mean(accs)}
def addCodeMaskToCalcuCodeRepr(model,code, relative_par_ids, relative_bro_ids, semantic_ids):
    relative_par_mask = relative_par_ids == 0
    relative_bro_mask = relative_bro_ids == 0
    semantic_mask = semantic_ids == 0
    code_mask = relative_mask([relative_par_mask, relative_bro_mask, semantic_mask], 6)
    code_repr = model.code_encoding(code, relative_par_ids, relative_bro_ids, semantic_ids, code_mask)
    return code_repr
def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
   
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--visual',default=False, help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=100, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
        
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

     # Model Hyperparameters for automl tuning
    #parser.add_argument('--emb_size', type=int, default=-1, help = 'embedding dim')
    parser.add_argument('--n_hidden', type=int, default= -1, help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default= -1)         
    parser.add_argument('--margin', type=float, default= -1)
    parser.add_argument('--sim_measure', type=str, default = 'cos', help='similarity measure for training')
    
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    #parser.add_argument('--adam_epsilon', type=float)
    #parser.add_argument("--weight_decay", type=float, help="Weight deay if we apply some.")
    #parser.add_argument('--warmup_steps', type=int)
    
    # reserved args for automl pbt
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)

    ##############################################################################################################################
    #parser = argparse.ArgumentParser(description='tree transformer')
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    i2code = read_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = read_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = read_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}
    # make output directory if it doesn't already exist
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)
    
    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn
   
    train(args,ast2id,code2id,nl2id,i2nl)
        
    