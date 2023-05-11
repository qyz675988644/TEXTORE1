import pandas as pd
import numpy as np
root = "/home/qiuyuanzhe/TEXTORE-main/frontend/weight_cache/results/"
save_root = "/home/qiuyuanzhe/TEXTORE-main/frontend/results"

def by_col(df:pd.DataFrame, col, val):
    res = df.groupby(col)
    res = res.get_group(val)
    # print(res)
    return res

def get_mean_from_df(data, m, d):
    res = []
    k = ['known_cls_ratio', 'Seen', 'Unseen', 'Overall', 'Acc']
    for name, df in data.groupby('known_cls_ratio'):
        res.append(df.agg({x:np.mean for x in k}))
    this_data = pd.DataFrame(res, columns=k)
    this_data.insert(loc=0, column='method', value=m, allow_duplicates=True)
    this_data.insert(loc=0, column='dataset', value=d, allow_duplicates=True)
    return this_data

def get_res(methods, datasets, wk=None):
    this_df = None
    for m in methods:
        for d in datasets:
            p = root + '/relation_detection/{}/{}_results.csv'.format(m, d)
            data = pd.read_csv(p)
            if this_df is None:
                this_df = get_mean_from_df(data, m, d)
            else:
                this_df = this_df.append(get_mean_from_df(data, m, d), ignore_index=True)
    this_df.to_csv(save_root + "/relation_detection/detection.csv", index=False)
    return this_df  


def get_mean_from_df_discovery(data, m, d):
    res = []
    std = []
    k = ['known_cls_ratio', 'ACC', 'ARI', 'NMI', 'B3']
    for name, df in data.groupby('known_cls_ratio'):
        res.append(df.agg({x:np.mean for x in k}) )
        std.append(df.agg({x:np.std for x in k}) )
    this_data = pd.DataFrame(res, columns=k)
    std_data = pd.DataFrame(std, columns=k)
    # for kk in k[1:]:
    #     this_data.loc[:, kk] = ["{}±{}".format(round(x, 2),round(y,2)) for x,y in zip(this_data.loc[:, kk].values.tolist(), std_data.loc[:, kk].values.tolist())]
    this_data.insert(loc=0, column='method', value=m, allow_duplicates=True)
    this_data.insert(loc=0, column='dataset', value=d, allow_duplicates=True)
    return this_data

def get_res_discovery(methods, datasets, wk=None):
    this_df = None
    for m in methods:
        for d in datasets:
            p = root + '/relation_discovery/{}/{}_results.csv'.format(m, d)
            data = pd.read_csv(p)
            if this_df is None:
                this_df = get_mean_from_df_discovery(data, m, d)
            else:
                this_df = this_df.append(get_mean_from_df_discovery(data, m, d), ignore_index=True)
    this_df.to_csv(save_root + "/relation_discover/discovery.csv", index=False)
    return this_df
def get_res_pipeline(p, d):
    this_df = None
    data = pd.read_csv(p)
    data['method'] = [dt+'_'+ds  for dt, ds in zip(data.loc[:, 'det_method'].values.tolist(), data.loc[:, 'dis_method'].values.tolist())]
    for m, m_df in data.groupby('method'):
        res = []
        std = []
        k = ['known_cls_ratio', 'ACC', 'ARI', 'NMI', 'B3', 'Known_Acc', 'Known_F1']
        for name, df in m_df.groupby('known_cls_ratio'):
            res.append(df.agg({x:np.mean for x in k}) )
            std.append(df.agg({x:np.std for x in k}) )
        this_data = pd.DataFrame(res, columns=k)
        std_data = pd.DataFrame(std, columns=k)
        # for kk in k[1:]:
        #     this_data.loc[:, kk] = ["{}±{}".format(round(x, 2),round(y,2)) for x,y in zip(this_data.loc[:, kk].values.tolist(), std_data.loc[:, kk].values.tolist())]
        this_data.insert(loc=0, column='method', value=m, allow_duplicates=True)
        this_data.insert(loc=0, column='dataset', value=d, allow_duplicates=True)
        if this_df is None:
            this_df = this_data
        else:
            this_df = this_df.append(this_data, ignore_index=True)
    this_df.to_csv("results/pipeline/pipe_{}.csv".format(d), index=False)
    return this_df
if __name__=="__main__":
    print("detection results:")
    print(get_res(['MSP', 'OpenMax', 'DOC', 'DeepUnk', 'ADB'], ['semeval', 'wiki80']) )
    print("discovery results:")
    print(get_res_discovery(['MORE', 'SelfORE', 'RSN'], ['semeval', 'wiki80']))
    