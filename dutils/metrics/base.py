
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np

class CorMetric(object):
    # update two elements
    def __init__(self):
        self.data = []

    def update(self, a, b):
        self.data.append([a, b])
    
    def decode(self):
        data = np.array(self.data)
        a = data[:, 0]
        b = data[:, 1]
        return a, b

    def spearman(self):
        a, b = self.decode()
        cor, pvalue = spearmanr(a, b)
        return cor, pvalue
    
    def pearsonr(self):
        a, b = self.decode()
        cor, conf = spearmanr(a, b)
        return cor, conf
    
    def condtion(self, idx=-1, min_=0, max_=1):
        a, b = self.decode()
        cond_ele = np.array(self.data)[:, idx]
        cond_mask = np.logical_and(cond_ele >= min_, cond_ele < max_)
        a = a[cond_mask]
        b = b[cond_mask]
        return a, b
    
    def reset(self):
        self.data = []

    def summarize(self, idx=None, min_=None, max_=None):
        if idx is not None:
            assert (min_ is not None) and (max_ is not None), "set wrong args for condition"
            a, b = self.condtion(idx, min_, max_)
        else:
            a, b = self.decode()
            
        scor, _ = spearmanr(a, b)
        pcor, _ = pearsonr(a, b)
        kt, _   = kendalltau(a, b)

        print("spearman cor:", scor)
        print("pearson cor:", pcor)
        print("kendall rank:", kt)


# alias
spearman = spearmanr
pearson = pearsonr
kendall = kendalltau