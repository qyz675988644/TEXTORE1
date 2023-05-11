from enum import Flag
import json, os, sys
import pickle
import random
import importlib
import pandas as pd
from copy import deepcopy
import copy
import yaml
import torch
from torch.nn import Parameter
from torch import autograd, optim, nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Bernoulli, kl_divergence
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, \
    RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import get_constant_schedule, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import sklearn.metrics
import logging
import json
from logging import handlers
import time
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from .metrics import *
from .backbone.bert_encoder import SentenceEncoder, SentenceEncoderWithHT#, BertEncoder
from .backbone.cnn_encoder import CNNSentenceEncoder
from keybert import KeyBERT
import networkx as nx
import community

class GetRelName:
    def __init__(self, args):
        self.logger = args.logger
        self.test_examples = load_json(os.path.join(args.data_dir,args.dataname,"test.json"))
        setup_seed(args.seed)
        self.logger.info('Loading KeyBERT model start...')
        self.keywords_model = KeyBERT('distilbert-base-nli-mean-tokens')
        self.logger.info('Loading KeyBERT model finished...')

    def get_keyword(self, test_examples, pred):
        test_examples = []
        for item in self.test_examples:
            token = item["token"]
            token = " ".join(token)
            test_examples.append(token)
        unique_preds = np.unique(np.array(pred))
        res = {}
        for unique_pred in unique_preds:
            label_sample_ids = [idx for idx, elem in enumerate(pred) if elem == unique_pred]
            label_texts = np.array(test_examples)[label_sample_ids]
            doc = " ".join([item for item in label_texts])
            keywords = self.keywords_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n = 3)
            res[unique_pred] = keywords
        return res


###############################################################
#######################  SaveData     #########################
###############################################################

class SaveData(object):
    def __init__(self, args, save_cfg =True) -> None:
        try:
            self.mid_dir = args.res_mid_dir
            self.path = args.res_path
        except:
            self.mid_dir = './'
            self.path = './'
        if save_cfg:
            save_yaml(self.path.replace('pkl', 'yaml'), args.__dict__)
            self.init_data()
        self.need_keyword = args.need_keyword
        if args.task_type == 'relation_discovery':
            if args.need_keyword:
                self.grn = GetRelName(args)
        else:
            self.need_keyword = 0


    def init_data(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.results = None
        self.samples = None
        self.features = None
        self.reduce_feat = None
        self.pred = None
        self.labels = None
        self.known_label_list = None
        self.all_label_list = None
        self.outputs = None
    def append_train_loss(self, x):
        self.append_data(self.train_loss, x)
    def append_val_loss(self, x):
        self.append_data(self.val_loss, x)
    def append_train_acc(self, x):
        self.append_data(self.train_acc, x)
    def append_val_acc(self, x):
        self.append_data(self.val_acc, x)
    def append_data(self, ls:list, x):
        if isinstance(x, float) or isinstance(x, int):
            ls.append(x)
        elif isinstance(x, torch.Tensor):
            ls.append(x.item())
        else:
            ls.append(x)
            # assert 0, "errors data: {}".format(x)
    def save_middle(self, res:dict = None):
        cur = {"train_loss": self.train_loss, "train_acc": self.train_acc, "val_loss":self.val_loss, "val_acc":self.val_acc}
        if res is not None:
            dictMerged = res.copy()
            dictMerged.update(cur)
        else:
            dictMerged = cur
        self.save_data(dictMerged)
    def save_data(self, data:dict):
        if os.path.isfile(self.path):
            this_data = load_pickle(self.path)
            dictMerged = this_data.copy()
            dictMerged.update(data)
        else:
            dictMerged = data
        save_pickle(self.path, dictMerged)
    
    def save_output_results(self, data:dict=None):
        self.outputs = {
            "samples": self.samples,
            "labels": self.labels,
            "pred": self.pred,
            "features": self.features,
            "reduce_feat": self.reduce_feat,
            "results": self.results,
            "train_loss": self.train_loss, "train_acc": self.train_acc, "val_loss":self.val_loss, "val_acc":self.val_acc,
            "known_label_list": self.known_label_list,
            "all_label_list": self.all_label_list,
            "keywords": self.grn.get_keyword(self.samples, self.pred) if self.need_keyword else 'none'
            
        }
        if data is not None:
            self.outputs.update(data)
        if os.path.isfile(self.path):
            this_data = load_pickle(self.path)
            dictMerged = this_data.copy()
            dictMerged.update(self.outputs)
        else:
            dictMerged = self.outputs
        
        save_pickle(self.path, dictMerged)
    
    def save_results(self, args, results:dict, use_thisname = False, save_path = None):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        if use_thisname:
            result_file = '{}_{}_results.csv'.format(args.dataname, args.this_name)
        else:
            result_file = '{}_results.csv'.format(args.dataname)
        if save_path is None:
            
            values = [now, args.method, args.dataname, args.this_name, args.lr, args.seed, args.known_cls_ratio, args.labeled_ratio] + list(results.values())
            keys = ['time', 'method', 'dataname', 'model_name', 'lr', 'seed', 'known_cls_ratio', 'label_ratio'] + list(results.keys())
            results = {k:values[i] for i, k in enumerate(keys)}
            results_path = os.path.join(self.mid_dir, result_file)
        else:
            # results_path = save_path
            results_path = os.path.join(save_path, result_file)
            values = [now, args.detection_method, args.discovery_method, args.dataname, args.this_name, args.lr, args.seed, args.known_cls_ratio, args.labeled_ratio] + list(results.values())
            keys = ['time', 'det_method', 'dis_method', 'dataname', 'this_name', 'lr', 'seed', 'known_cls_ratio', 'label_ratio'] + list(results.keys())
            results = {k:values[i] for i, k in enumerate(keys)}
        args.logger.info("save final results: {}".format(results_path))
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)
        args.logger.info("save final results: {}".format(results))
    

###############################################################
#######################  Manager     ##########################
###############################################################
class BaseManager:
    def __init__(self, args):
        self.logger = args.logger
        if torch.cuda.is_available() and args.gpu_id != -1:
            self.device = torch.device("cuda")
            self.logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            self.logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
        else:
            self.logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.saver = SaveData(args)
        self.global_step = 0
        self.lr = args.lr
    def warmup(self, optimizer, warmup_step=500):
        self.global_step += 1
        if self.global_step < warmup_step:
            warmup_rate = float(self.global_step) / warmup_step
        else:
            warmup_rate = 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr * warmup_rate
    def freeze_bert_parameters(self, model):
        for name, param in model.encoder.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def to_cuda(self, *args):
        output = []
        for ins in args:
            if isinstance(ins, dict):
                for k in ins:
                    ins[k] = ins[k].to(self.device)
            elif isinstance(ins, torch.Tensor):
                ins = ins.to(self.device)
            output.append(ins)
        if len(args)==1:
            return ins
        return output

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    
    def get_optimizer(self, args, model):
        print('Use {} optim!'.format(args.optim))
        if args.optim == 'adamw':
            
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(parameters_to_optimize, lr=args.lr, correct_bias=False)
        else:
            
            if args.optim == 'sgd':
                # pytorch_optim = optim.SGD
                optimizer = optim.SGD(
                model.parameters(),
                args.lr, 
                weight_decay=1e-5,
                momentum=0.9
            )
                return optimizer
            elif args.optim == 'adam':
                pytorch_optim = optim.Adam
            elif args.optim == 'rmp':
                pytorch_optim = optim.RMSprop
            else:
                raise NotImplementedError
            optimizer = pytorch_optim(
                model.parameters(),
                args.lr, 
                weight_decay=args.weight_decay
            )
        return optimizer
    def get_scheduler(self, args, optimizer, warmup_step=None, train_iter=None):
        args.logger.info('Use {} schedule!'.format(args.sche))
        if args.sche == 'linear_warmup':
            args.logger.info("warmup_step: {}".format(warmup_step))
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = warmup_step,
                num_training_steps = train_iter
            )
        elif args.sche == 'const_warmup':
            args.logger.info("warmup_step: {}".format(warmup_step))
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                args.warmup_step
            )
        elif args.sche == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size)
        else:
            raise NotImplementedError
        return scheduler
    def fp16(self, model, optimizer):
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        return model, optimizer

def get_manager(args, data=None):
    if args.task_type in ['relation_discovery']:
        module_names = [args.task_type, 'methods', args.method_type, args.method, 'manager']
    else:
        module_names = [args.task_type, 'methods', args.method, 'manager']
    import_name = ".".join(module_names)
    method = importlib.import_module(import_name)
    if data is not None:
        manager = method.Manager(args, data)
    else:
        manager = method.Manager(args)
    return manager

def load_model(ckpt, k=0):
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage.cuda(k))
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)

def restore_model(args, output_model_file=None):
    if output_model_file is None:
        output_model_file = args.output_model_file
    if not os.path.isfile(output_model_file):
        assert 0, "{} is not a torch file!".format(output_model_file)
    ckpt = load_model(output_model_file, args.gpu_id)
    args.logger.info("Successfully loaded checkpoint '%s'" % output_model_file)
    return ckpt

def save_model(args, save_dict):
    args.logger.info("save best ckpt!")
    args.logger.info("save file: {}".format(args.output_model_file))
    torch.save(save_dict, args.output_model_file)

def save_pretrain_model(args, save_dict):
    args.logger.info("save best ckpt!")
    args.logger.info("save file: {}".format(args.pretrain_model_file))
    torch.save(save_dict, args.pretrain_model_file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dimension_reduction(features, seed):
    tsne = TSNE(n_components=2, metric="cosine", init="random", random_state=seed) # metric="euclidean" "cosine"
    result = tsne.fit_transform(features)
    return result


def get_logger(path, name:str):
    rq = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = os.path.join(path, 'open_re_{}.log'.format(rq))
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    con_handler = handlers.TimedRotatingFileHandler(filename=path, when='D', encoding='utf-8')
    con_handler.formatter = formatter
    con_handler.setLevel(logging.INFO)
    logger.addHandler(con_handler)

    hh = logging.StreamHandler(sys.stdout)
    hh.formatter = formatter
    hh.setLevel(logging.INFO)
    logger.addHandler(hh)
    logger.setLevel(logging.INFO)
    return logger


def round_dict(a:dict):
    for k,v in a.items():
        a[k] = round(v*100, 2)
    return a
def save_json(path: str, d):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4)
def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d
def save_pickle(path:str, d):
    with open(path, 'wb') as f:
        pickle.dump(d, f)
def load_pickle(path:str):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

def save_yaml(path:str, d:dict):
    this_d = {}
    for k, v in d.items():
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            this_d[k] = v
    with open(path, 'w') as file: 
        yaml.dump(this_d, file, default_flow_style=False)

def load_yaml(path:str, args=None):
    path = "/home/qiuyuanzhe/TEXTORE-main/{}".format(path)
    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
        if args is not None:
            for key, value in d.items():
                vars(args)[key] = value
    return d

def creat_check_path(root, *arg):
    mid_dir = root
    for temp_dir in arg:
        mid_dir = os.path.join(mid_dir, temp_dir)
        if not os.path.exists(mid_dir):
            os.makedirs(mid_dir)
    return mid_dir

## clustering
def Louvain_no_isolation(dataset, edge_measure, datatype=np.int32, iso_thres=5):

    print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i,j))

    print('calculating edges...')
    batch_count = 0
    batch_size = 10000
    left_data = []
    right_data = []
    edge_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data.append(dataset[idx_pair[0]])
        right_data.append(dataset[idx_pair[1]])
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...',str(round(count/len(idx_list)*100,2))+'%', end='')
            temp_edge_list = edge_measure(left_data, right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
            left_data = []
            right_data = []
    if batch_count !=0:
        print('predicting...')
        temp_edge_list = edge_measure(left_data, right_data)
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    simi_list = edge_list
    edge_list = np.int32(np.round(edge_list))

    #------------------
    print('forming simi_matrix...')
    simi_matrix = np.zeros([len(dataset),len(dataset)])
    for count,idx_pair in enumerate(idx_list):
        simi_matrix[idx_pair[0],idx_pair[1]] = simi_list[count]
        simi_matrix[idx_pair[1],idx_pair[0]] = simi_list[count]
    #------------------

    print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if edge_list[i]==0:
            true_edge_list.append(idx_list[i])
    g.add_edges_from(true_edge_list)

    print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    print('decoding to get label_list...')
    label_list = [0]*len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    #------------------
    print('solving isolation...')
    cluster_datanum_dict = {}
    for reltype in label_list:
        if reltype in cluster_datanum_dict.keys():
            cluster_datanum_dict[reltype] += 1
        else:
            cluster_datanum_dict[reltype] = 1


    iso_reltype_list = []
    for reltype in cluster_datanum_dict:
        if cluster_datanum_dict[reltype]<=iso_thres:
            iso_reltype_list.append(reltype)

    for point_idx, reltype in enumerate(label_list):
        if reltype in iso_reltype_list:
            search_idx_list = np.argsort(simi_matrix[point_idx]) # from small to big
            for idx in search_idx_list:
                if label_list[idx] not in iso_reltype_list:
                    label_list[point_idx] = label_list[idx]
                    break
    #------------------

    return label_list

# find the closest two classes
def find_close(M):
    s_index, l_index = 0, 0
    min_list = np.zeros([len(M)],dtype=np.float32)
    min_index_list = np.zeros([len(M)],dtype=np.int32)
    for i,item in enumerate(M):
        if len(item):
            temp_min = min(item)
            min_list[i] = temp_min
            min_index_list[i] = item.index(temp_min)
        else:
            min_list[i] = 10000
    l_index = int(np.where(min_list==np.min(min_list))[0][0])
    s_index = min_index_list[l_index]
    return s_index, l_index # s_index < l_index
def complete_HAC(dataset, HAC_dist, k, datatype=np.int32):
    #initialize C and M, C is a list of clusters, M is a list as dist_matrix

    print('the len of dataset to cluster is:'+str(len(dataset)))
    print('initializing...')
    idx_C, M, idxM = [], [], []
    for i,item in enumerate(dataset):
        idx_Ci = [i]
        idx_C.append(idx_Ci)

    print('initializing dist_matrix...')
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(idx_C)):
        for j in range(len(idx_C)):
            if j == i:
                break
            idx_list.append([i,j])

    print('calculating dist_list...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    dist_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data[batch_count]=dataset[idx_pair[0]]
        right_data[batch_count]=dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting',str(round(count/len(idx_list)*100,2))+'%')
            temp_dist_list = HAC_dist(left_data,right_data)
            dist_list = dist_list + temp_dist_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count !=0:
        print('predicting...')
        temp_dist_list = HAC_dist(left_data[:batch_count],right_data[:batch_count])
        dist_list = dist_list + temp_dist_list.reshape(batch_count).tolist()

    print('preparing dist_matrix...')
    count = 0
    for i in range(len(idx_C)):
        Mi = []
        for j in range(len(idx_C)):
            if j == i:
                break
            Mi.append(dist_list[count])
            count += 1
        M.append(Mi)

    # combine two classes
    q = len(idx_C)
    while q > k:
        s_index, l_index = find_close(M)
        idx_C[s_index].extend(idx_C[l_index])
        del idx_C[l_index]

        M_next = deepcopy(M[:-1])
        for i in range(len(idx_C)):
            for j in range(len(idx_C)):
                if j == i:
                    break

                i_old, j_old = i, j
                if i >= l_index:
                    i_old = i + 1
                if j >= l_index:
                    j_old = j + 1

                if i != s_index and j != s_index:
                    M_next[i][j]=M[i_old][j_old]
                elif i == s_index:
                    M_next[i][j]=max(M[s_index][j_old],M[l_index][j_old])
                elif j == s_index:
                    if i_old<l_index:
                        M_next[i][j]=max(M[i_old][s_index],M[l_index][i_old])
                    elif i_old>l_index:
                        M_next[i][j]=max(M[i_old][s_index],M[i_old][l_index])
        q -= 1
        print('temp cluster num is:',q,',',s_index,'and',l_index,'are combined, metric is:',M[l_index][s_index])
        M = M_next

    # decode to get label_list
    label_list = [0]*len(dataset)
    for label, temp_cluster in enumerate(idx_C):
        for idx in temp_cluster:
            label_list[idx] = label

    return label_list

def k_means(data, clusters): # range from 0 to clusters - 1
    #return KMeans(n_clusters=clusters,random_state=0,algorithm='full').fit(data).predict(data)
    return KMeans(n_clusters=clusters,algorithm='full').fit(data).predict(data)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def cal_info(pseudo_label_list, true_label):
    """
    pseudo_label_list: 二维列表
    true_label: 一维numpy
    """
    p_acc_list = []
    num_pseudo_label = len(pseudo_label_list)
    NMI = np.zeros((num_pseudo_label, num_pseudo_label))
    for pseudo_label in pseudo_label_list:
        p_acc_list.append(eval_acc(true_label, maplabels(true_label, np.array(pseudo_label))))
    for i in range(num_pseudo_label):
        for j in range(num_pseudo_label):
            if i == j:
                continue
            try:
                NMI[i][j] = metrics.normalized_mutual_info_score(np.array(pseudo_label_list[i]),np.array(pseudo_label_list[j]))
            except:
                print(i,j)
                exit()
    mean_NMI = np.mean(NMI, axis = 1).tolist()
    data = [(acc, nmi) for acc, nmi in zip(p_acc_list, mean_NMI)]
    data.sort()
    p_acc_list = [acc for acc, _ in data]
    mean_NMI = [nmi for _, nmi in data]
    print("####")
    print(p_acc_list)
    print(mean_NMI)
    print("####")

def compute_kld2(p_logit, q_logit):
    p = F.softmax(p_logit, dim = 1) # (B, n_class) 
    q = F.softmax(q_logit, dim = 1) # (B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = 1)
    
def compute_kld(p_logit, q_logit):
    p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
    q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)
    
def data_split(examples, ratio = [0.6, 0.2, 0.2]):
    from sklearn.utils import shuffle
    examples = shuffle(examples)
    train_num = int(len(examples) * ratio[0])
    dev_num = int(len(examples) * ratio[1])
    train_examples = examples[:train_num]
    dev_examples = examples[train_num:train_num + dev_num]
    test_examples = examples[train_num + dev_num:]
    return train_examples, dev_examples, test_examples

def data_split2(examples, ratio = [6/7, 1/7]):  #分开训练和测试
    from sklearn.utils import shuffle   #打乱样本
    examples = shuffle(examples)
    dev_num = int(len(examples) * ratio[0])     #6/7做训练
    dev_examples = examples[:dev_num]
    test_examples = examples[dev_num:]
    return dev_examples, test_examples

def L2Reg(net):
    reg_loss = 0
    for name, params in net.named_parameters():
        if name[-4:] != 'bias':
            reg_loss += torch.sum(torch.pow(params, 2))
    return reg_loss

def flush():#清下gpu
    '''
    :rely on:import torch
    '''
    torch.cuda.empty_cache()
    '''
    或者在命令行中
    ps aux | grep python
    kill -9 [pid]
    或者
    nvidia-smi --gpu-reset -i [gpu_id]
    '''
    
def clean_text(text): # input is word list
    ret = []
    for word in text:   #text里面的单词
        normalized_word = re.sub(u"([^\u0020-\u007f])", "", word)   #特殊字符用“”替代
        if normalized_word == '' or normalized_word == ' ' or normalized_word == '    ':
            normalized_word = '[UNK]'   #空白word表示为[UNK]
        ret.append(normalized_word)     #剩下的单词存到列表里
    return ret
    
def get_pseudo_label(pseudo_label_list, idxes):    
    ret = []
    for idx in idxes:
        ret.append(pseudo_label_list[idx])
    ret = torch.tensor(ret).long()
    return ret


def eval_acc(L1, L2):
    sum = np.sum(L1[:]==L2[:])
    return sum/len(L2)

def eval_p_r_f1(ground_truth, label_pred):
    def _bcubed(i):
        C = label_pred[i]
        n = 0
        for j in range(len(label_pred)):
            if label_pred[j] != C:
                continue
            if ground_truth[i] == ground_truth[j]:
                n += 1
        p = n / num_cluster[C]
        r = n / num_class[ground_truth[i]]
        return p, r

    ground_truth -= 1
    label_pred -= 1
    num_class = [0] * 16
    num_cluster = [0] * 16
    for c in ground_truth:
        num_class[c] += 1
    for c in label_pred:
        num_cluster[c] += 1

    precision = recall = fscore = 0.

    for i in range(len(label_pred)):
        p, r = _bcubed(i)
        precision += p
        recall += r
    precision = precision / len(label_pred)
    recall = recall / len(label_pred)
    fscore = 2 * precision * recall / (precision + recall) # f1 score
    return precision, recall, fscore

def tsne_and_save_data(data, label, epoch):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import numpy as np
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    #fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    f = "./{}.npz".format(epoch)
    np.savez(f, x=x, y=y, true_label=label)

def tsne_and_save_pic(data, label, new_class, acc):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    '''
    m = {1:'o',2:'s',3:'D',4:'+'}
    cm = list(map(lambda x:m[x],c))#将相应的标签改为对应的marker
    '''
    #scatter = mscatter(x, y, c=c, m=cm, ax=ax,cmap=plt.cm.RdYlBu)
    scatter = mscatter(x, y, c=c, ax=ax,cmap=plt.cm.Paired, s = 9.0)
    if new_class:
        ax.set_title("t-sne for new class, result:{}".format(acc))
        plt.savefig( 't-sne-new-class.svg' ) 
    else:
        ax.set_title("t-sne for known class, result:{}".format(acc))
        plt.savefig( 't-sne-known-class.svg' ) 

def maplabels(L1, L2):
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    row_ind, col_ind = linear_sum_assignment(-G.T)
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        for j in range(len(L2)):
            if L2[j] == row_ind[i]:
                newL2[j] = col_ind[i]
    return newL2

def endless_get_next_batch(loaders, iters, batch_size):
    try:
        data = next(iters)
    except StopIteration:
        iters = iter(loaders)
        data = next(iters)
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size
    if len(data[0]) < batch_size:
        return endless_get_next_batch(loaders, iters, batch_size)
    return data, iters

def _worker_init_fn_():
    import random
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // (2**32-1)
    random.seed(torch_seed)
    np.random.seed(np_seed)