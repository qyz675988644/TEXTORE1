from tools.utils import *

def update_centers_l(net, args, known_class_dataloader):
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    centers = torch.zeros(args.num_class, args.kmeans_dim, device = device)
    num_samples = [0] * args.num_class
    with torch.no_grad():
        for iteration, (input_ids, input_mask, label, head_span, tail_span) in enumerate(known_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, label, head_span, tail_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, kmeans_dim)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label[i]
                centers[l] += vec
                num_samples[l] += 1
        for c in range(args.num_class):
            centers[c] /= num_samples[c]
        net.module.ct_loss_l.centers = centers.to(device)

def update_centers_u(net, args, new_class_dataloader):
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=args.new_class,random_state=0,algorithm='full')
    true = [-1] * len(new_class_dataloader.dataset)
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        rep = []
        idxes = []
        true_label = []
        for iteration, (input_ids, input_mask, label, head_span, tail_span, idx, _) in enumerate(new_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, label, head_span, tail_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, kmeans_dim)
            true_label.append(label)
            idxes.append(idx)
            rep.append(sia_rep)
        rep = torch.cat(rep, dim = 0).cpu().numpy() # (num_test_ins, kmeans_dim)
        #cat 在给定维度上对rep进行操作，将变量放在cpu上，将tensor转换为numpy
        idxes = torch.cat(idxes, dim = 0).cpu().numpy()
        true_label = torch.cat(true_label, dim = 0).cpu().numpy()

    label_pred = clf.fit_predict(rep)# from 0 to args.new_class - 1 返回每个数据对应的标签，并将标签值对应到相应的簇
    net.module.ct_loss_u.centers = torch.from_numpy(clf.cluster_centers_).to(device) # (num_class, kmeans_dim)
    for i in range(len(idxes)):
        idx = idxes[i]
        pseudo = label_pred[i]
        true[idx] = true_label[i]
        new_class_dataloader.dataset.examples[idx].pseudo = pseudo
        #pseudo_label_list[cnt][idx] = pseudo
