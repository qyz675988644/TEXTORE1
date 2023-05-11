from tools.utils import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tools.utils import *
import math
from .model import *
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)


# 先训练AUTOENCODER 10轮，再一起训
def load_pretrained(args):
    from transformers import BertTokenizer, BertModel, BertConfig
    config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model, config = config)
    return config, tokenizer, bert

class PretrainManager(BaseManager):
    """
    from https://github.com/thuiar/TEXTOIR
    """
    def __init__(self, args, data=None):
        super(PretrainManager, self).__init__(args)
        self.model = ZeroShotModel3(args).to(self.device)
        self.optimizer = self.get_optimizer(args, self.model)
        num_train_examples = len(data.train_feat)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.scheduler = self.get_scheduler(
            args, 
            self.optimizer, 
            warmup_step=args.warmup_proportion * num_train_optimization_steps,
            train_iter=num_train_optimization_steps
        )
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.train_dataloader = data.sup_train_dataloader
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader 
        self.loss_fct = nn.CrossEntropyLoss()

    def train(self, args, data):
        wait = 0
        best_model = None
        best_eval_score = 0
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                input_ids, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)
                # batch = tuple(t.to(self.device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):

                    loss = self.model(input_ids, batch_mask, batch_pos, label_ids, mode = "train", loss_fct = self.loss_fct)
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                self.save_model(args)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break
                
    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):
            input_ids, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)
            # batch = tuple(t.to(self.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, batch_mask, batch_pos)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        # save_pretrain_model()
        save_model(args, save_dict)

class Manager(BaseManager):
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        self.model = ZeroShotModel3(args).to(self.device)
        self.optimizer = self.get_optimizer(args, self.model)
        num_train_examples =len(data.semi_feat)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.scheduler = self.get_scheduler(
            args,
            self.optimizer, 
            warmup_step=args.warmup_proportion * num_train_optimization_steps,
            train_iter=num_train_optimization_steps
        )
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.train_labeled_dataloader = data.sup_train_dataloader
        self.train_unlabeled_dataloader = data.train_dataloader
        self.train_dataloader = data.semi_train_dataloader
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader 
        self.config, self.tokenizer, self.pretrained_model = load_pretrained(args)

        if args.train_model:
            pretrain_manager = PretrainManager(args, data)
            self.logger.info('Pre-raining start...')
            pretrain_manager.train(args, data)
            self.logger.info('Pre-training finished...')

            self.centroids = None
            self.pretrained_model = pretrain_manager.model

            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data) 
            else:
                self.num_labels = data.num_labels 
            self.restore_model(args)
            # self.load_pretrained_model(self.pretrained_model)
        else:
            # self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.restore_model(args)
            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data) 
            else:
                self.num_labels = data.num_labels 
    
    def train(self, args, epoch, known_class_dataloader, new_class_dataloader, optimizer):
        self.model.train()
        self.known_class_iter = iter(known_class_dataloader)
        self.new_class_iter = iter(new_class_dataloader)
        siamese_known_class_iter = iter(known_class_dataloader)
        siamese_new_class_iter = iter(new_class_dataloader)
        if args.IL:
            icr_new_class_iter = iter(new_class_dataloader)
            icr_known_class_iter = iter(known_class_dataloader)

        epoch_ce_loss = 0
        epoch_ct_loss = 0
        epoch_rec_loss = 0
        epoch_u_loss = 0
        epoch_acc = 0 

        with tqdm(total=len(known_class_dataloader), desc='training') as pbar:
            for iteration in range(len(known_class_dataloader)):
                self.optimizer.zero_grad()
    # 一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
                # Training unlabeled head
                if epoch > args.num_pretrain:
                    # Training unlabeled head
                    data, new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = new_class_iter, batch_size = args.b_size)
                    data, pseudo_label = data[:-2], data[-1]
                    # pretrained_feat = net.forward(data, msg = 'feat').detach() # (batch_size, new_class)
                    batch_size = pseudo_label.size(0)
                    # ground_truth = data[2] - args.num_class
                    pseudo_label = (pseudo_label.unsqueeze(0) == pseudo_label.unsqueeze(1)).float()  #unqueeze：dim = dim+input.dim()+1
                    logits = self.model.forward(data, msg = 'unlabeled', cut_gradient = False) # (batch_size, new_class)
                    expanded_logits = logits.expand(batch_size, -1, -1)
                    expanded_logits2 = expanded_logits.transpose(0, 1)
                    kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
                    kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)
                    assert kl1.requires_grad
                    u_loss = torch.mean(pseudo_label * (kl1 + kl2) + (1 - pseudo_label) * (torch.relu(args.sigmoid - kl1) + torch.relu(args.sigmoid - kl2)))
                    #无标签数据，同类，不同类
                    u_loss.backward()
                    #flush()
                else:
                    u_loss = torch.tensor(0)

                # Training siamese head (exclude bert layer)
                data, siamese_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = siamese_known_class_iter, batch_size = args.b_size)
                sia_rep1, rec_loss1 = self.model.forward(data, msg = 'reconstruct') # (batch_size, kmeans_dim)
                label = data[2]
                data, siamese_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = siamese_new_class_iter, batch_size = args.b_size)
                data, pseudo = data[:-2], data[-1]
                sia_rep2, rec_loss2 = self.model.forward(data, msg = 'reconstruct') # (batch_size, kmeans_dim)
                rec_loss = (rec_loss1.mean() + rec_loss2.mean()) / 2
                pseudo = pseudo
                label = label
                ct_loss = args.ct * self.module.ct_loss_l(label, sia_rep1)
                loss = rec_loss + ct_loss + 1e-5 * (L2Reg(self.module.similarity_encoder) + L2Reg(net.module.similarity_decoder))
                loss.backward()
                #flush() 
                self.saver.append_train_loss(loss)

                if not args.IL and epoch > args.num_pretrain:
                    # Training labeled head and specific layer of bert
                    data, known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = known_class_iter, batch_size = args.b_size)
                    known_logits = self.forward(data, msg = 'labeled')
                    if args.IL:
                        known_logits = known_logits[:,:args.num_class]
                    label_pred = torch.max(known_logits, dim = -1)[1]
                    known_label = data[2]
                    acc = 1.0 * torch.sum(label_pred == known_label) / len(label_pred)
                    ce_loss = self.module.ce_loss(input = known_logits, target = known_label)
                    ce_loss.backward()
                    self.saver.append_train_loss(ce_loss)
                    #flush()
                else:
                    ce_loss = torch.tensor(0)
                    acc = torch.tensor(0)
                if args.IL:
                    if epoch > args.num_pretrain:
                        data, icr_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = icr_new_class_iter, batch_size = args.b_size)
                        data = data[:-2]
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model.forward(data, msg = 'unlabeled')
                            u_label = torch.max(logits, dim = -1)[1] + args.num_class
                        self.model.train()
                        u_logits = self.model.forward(data, msg = 'labeled', cut_gradient = False)

                        data, icr_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = icr_known_class_iter, batch_size = args.b_size)
                        l_logits = self.model.forward(data, msg = 'labeled', cut_gradient = False)
                        l_label = data[2]
                        #icr_ce_loss = args.rampup_cof * sigmoid_rampup(epoch-args.num_pretrain, args.rampup_length) * net.module.ce_loss(input = logits, target = u_label)
                        icr_ce_loss = args.rampup_cof * sigmoid_rampup(epoch-args.num_pretrain, args.rampup_length) * self.module.ce_loss(input=u_logits, target=u_label) \
                                        + self.module.ce_loss(input=l_logits, target=l_label)
                        icr_ce_loss.backward()
                        #flush()
                    else:
                        icr_ce_loss = torch.tensor(0)  

                self.optimizer.step()  
                epoch_ce_loss += ce_loss.item()
                epoch_ct_loss += ct_loss.item()
                epoch_rec_loss += rec_loss.item()
                epoch_u_loss += u_loss.item()
                epoch_acc += acc.item()
                pbar.update(1)
                pbar.set_postfix({"acc":epoch_acc / (iteration + 1), "ce loss":epoch_ce_loss / (iteration + 1), "rec loss":epoch_rec_loss / (iteration + 1), "ct_loss":epoch_ct_loss / (iteration + 1), "u_loss":epoch_u_loss / (iteration + 1), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                self.saver.append_train_loss(self.train_loss)
        num_iteration = len(known_class_dataloader)
        print("===> Epoch {} Complete: Avg. ce Loss: {:.4f}, rec Loss: {:.4f}, ct Loss: {:.4f}, u Loss: {:.4f}, known class acc: {:.4f}".format(epoch, epoch_ce_loss / num_iteration, epoch_rec_loss / num_iteration, epoch_ct_loss / num_iteration, epoch_u_loss / num_iteration, epoch_acc / num_iteration))

    def test_one_epoch(self, args, new_class_dataloader):
        import random
        self.model.eval()
        self.new_class_dataloader = new_class_dataloader
        with torch.no_grad():
            ground_truth = []
            label_pred = []
            with tqdm(total=len(new_class_dataloader), desc='testing') as pbar:
                for iteration, data in enumerate(new_class_dataloader):
                    data = data[:-2]
                    logits = self.model.forward(data, msg = 'unlabeled')
                    ground_truth.append(data[2])
                    label_pred.append(logits.max(dim = -1)[1].cpu())
                    pbar.update(1)
                label_pred = torch.cat(label_pred, dim = 0).numpy()
                ground_truth = torch.cat(ground_truth, dim = 0).numpy() - args.num_class
                cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
                B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
        print("B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}".format(B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI))
        print(cluster_eval)
        return cluster_eval['F1']

    def test_mix(known_pred, known_truth, new_pred, new_truth):
        pred = np.concatenate([known_pred, new_pred])
        truth = np.concatenate([known_truth, new_truth])
        cluster_eval = ClusterEvaluation(truth,pred).printEvaluation()
        print("mix result:",cluster_eval)
        return cluster_eval['F1']

    def test_one_epoch2(self, args, epoch, new_class_dataloader, labelled = False):
        import random
        self.model.eval()
        desc = 'labelled' if labelled else 'unlabelled'
        with torch.no_grad():
            ground_truth = []
            label_pred = []
            with tqdm(total=len(new_class_dataloader), desc=desc) as pbar:
                for iteration, data in enumerate(new_class_dataloader):
                    if not labelled:
                        data = data[:-2]
                    logits = self.model.forward(data, msg = 'labeled')
                    ground_truth.append(data[2])
                    label_pred.append(logits.max(dim = -1)[1].cpu())
                    pbar.update(1)
                label_pred = torch.cat(label_pred, dim = 0).numpy()
                ground_truth = torch.cat(ground_truth, dim = 0).numpy()
                cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
        print(cluster_eval)
        #return cluster_eval['F1'], label_pred, ground_truth

        relation_info_dict = load_relation_info_dict(args, tokenizer) # mapping from relation name to rel id and description words

        known_class_train_examples = known_class_train_examples[:int(len(known_class_train_examples)*args.p)]   #可以截取训练样本数，默认为100%
        known_class_train_dataloader = DataLoader(BertDataset(args, known_class_train_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
        known_class_test_dataloader = DataLoader(BertDataset(args, known_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
        print("knwon class dataloader ready...")
        new_class_train_dataloader = DataLoader(PBertDataset(args, new_class_train_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
        new_class_test_dataloader = DataLoader(PBertDataset(args, new_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
        print("new class dataloader ready...")
        net = ZeroShotModel3(args, config, pretrained_model, unfreeze_layers = [args.layer])
        if args.cuda:
            net.cuda()
            net = nn.DataParallel(net)  #实现数据并行
        print("net ready...")
        print("-"*32)
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
        best_result = 0
        best_test_result = 0
        wait_times = 0
        for epoch in range(1, args.epochs + 1):
            print("\n-------EPOCH {}-------".format(epoch))
            if epoch > args.num_pretrain:   #epoch 100 num_pretrain 10
                wait_times += 1        
            update_centers_u(net, args, new_class_train_dataloader)
            update_centers_l(net, args, known_class_train_dataloader)
            train_one_epoch(net, args, epoch, known_class_train_dataloader, new_class_train_dataloader, optimizer)
            if args.IL:
                _, known_pred, known_truth = test_one_epoch2(net, args, epoch, known_class_test_dataloader, labelled = True)
                test_one_epoch2(net, args, epoch, new_class_train_dataloader, labelled = False)
                _, new_pred, new_truth = test_one_epoch2(net, args, epoch, new_class_test_dataloader, labelled = False)
                result = test_mix(known_pred, known_truth, new_pred, new_truth)
                test_result = result
                test_one_epoch(net, args, epoch, new_class_train_dataloader)
                test_one_epoch(net, args, epoch, new_class_test_dataloader)
            else:
                test_one_epoch2(net, args, epoch, known_class_test_dataloader, labelled = True)
                result = test_one_epoch(net, args, epoch, new_class_train_dataloader)
                test_result = test_one_epoch(net, args, epoch, new_class_test_dataloader)
            if result > best_result:
                wait_times = 0
                best_result = result
                best_test_result = test_result
                print("new class dev best result: {}, test result: {}".format(best_result, test_result))
                
            if wait_times > args.wait_times:
                print("wait times arrive: {}, stop training, best result is: {}".format(args.wait_times, best_test_result))
                break

    def contingency_matrix(ref_labels, sys_labels):
        """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
        from scipy.sparse import coo_matrix
        ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
        sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
        n_frames = ref_labels.size
        # Following works because coo_matrix sums duplicate entries. Is roughly
        # twice as fast as np.histogram2d.
        cmatrix = coo_matrix(
            (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
            shape=(ref_classes.size, sys_classes.size),
            dtype=np.int)
        cmatrix = cmatrix.toarray()
        return cmatrix, ref_classes, sys_classes

    def bcubed(ref_labels, sys_labels, cm=None):
        """Return B-cubed precision, recall, and F1.
        The B-cubed precision of an item is the proportion of items with its
        system label that share its reference label (Bagga and Baldwin, 1998).
        Similarly, the B-cubed recall of an item is the proportion of items
        with its reference label that share its system label. The overall B-cubed
        precision and recall, then, are the means of the precision and recall for
        each item.
        Parameters
        ----------
        ref_labels : ndarray, (n_frames,)
            Reference labels.
        sys_labels : ndarray, (n_frames,)
            System labels.
        cm : ndarray, (n_ref_classes, n_sys_classes)
            Contingency matrix between reference and system labelings. If None,
            will be computed automatically from ``ref_labels`` and ``sys_labels``.
            Otherwise, the given value will be used and ``ref_labels`` and
            ``sys_labels`` ignored.
            (Default: None)
        Returns
        -------
        precision : float
            B-cubed precision.
        recall : float
            B-cubed recall.
        f1 : float
            B-cubed F1.
        References
        ----------
        Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
        chains." Proceedings of LREC 1998.
        """
        ref_labels = np.array(ref_labels)
        sys_labels = np.array(sys_labels)
        if cm is None:
            cm, _, _ = contingency_matrix(ref_labels, sys_labels)
        cm = cm.astype('float64')
        cm_norm = cm / cm.sum()
        precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
        recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
        f1 = 2*(precision*recall)/(precision + recall)
        return precision, recall, f1

    def usoon_eval(label, pesudo_true_label):

        from sklearn.metrics.cluster import homogeneity_completeness_v_measure
        from sklearn.metrics import classification_report
        from sklearn.metrics.cluster import adjusted_rand_score
        
        ARI = adjusted_rand_score(label, pesudo_true_label)
        
        # res_dic = classification_report(label, pesudo_true_label,labels=name, output_dict=True)
        # return precision, recall, f1
        B3_prec, B3_rec, B3_f1 = bcubed(label, pesudo_true_label)
        # B3_f1 = res_dic["weighted avg"]['f1-score']
        # B3_prec = res_dic["weighted avg"]['precision']
        # B3_rec = res_dic["weighted avg"]['recall']
        
        v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(label, pesudo_true_label)
        return B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict)

class ClusterEvaluation:
    '''
    groundtruthlabels and predicted_clusters should be two list, for example:
    groundtruthlabels = [0, 0, 1, 1], that means the 0th and 1th data is in cluster 0,
    and the 2th and 3th data is in cluster 1
    '''
    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet = self.createGroundTruthSets(groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)

    def createGroundTruthSets(self, labels):

        groundtruthsets= {}
        assessableElems = set()

        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)

        return groundtruthsets, assessableElems

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalPrecision += self.b3precision(self.predictedsets[c],
                                                   self.findCluster(r, self.groundtruthsets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalRecall += self.b3recall(self.predictedsets[c], self.findCluster(r, self.groundtruthsets))

        return totalRecall / float(len(self.assessableElemSet))

    def findCluster(self, a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    def printEvaluation(self):

        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)

        m = {'F1': F1B3, 'F0.5': F05B3, 'precision': precB3, 'recall': recB3}
        return m

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3

class ClusterRidded:
    def __init__(self, gtlabels, prelabels, rid_thres=5):
        self.gtlabels = np.array(gtlabels)
        self.prelabels = np.array(prelabels)
        self.cluster_num_dict = {}
        for item in self.prelabels:
            temp = self.cluster_num_dict.setdefault(item, 0)
            self.cluster_num_dict[item] = temp + 1
        self.NA_list = np.ones(self.gtlabels.shape) # 0 for NA, 1 for not NA
        for i,item in enumerate(self.prelabels):
            if self.cluster_num_dict[item]<=rid_thres:
                self.NA_list[i] = 0
        self.gtlabels_ridded = []
        self.prelabels_ridded = []
        for i, item in enumerate(self.NA_list):
            if item==1:
                self.gtlabels_ridded.append(self.gtlabels[i])
                self.prelabels_ridded.append(self.prelabels[i])
        self.gtlabels_ridded = np.array(self.gtlabels_ridded)
        self.prelabels_ridded = np.array(self.prelabels_ridded)
        print('NA clusters ridded, NA num is:',self.gtlabels.shape[0]-self.gtlabels_ridded.shape[0])

    def printEvaluation(self):
        return ClusterEvaluation(self.gtlabels_ridded,self.prelabels_ridded).printEvaluation()

