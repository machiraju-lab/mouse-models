import glob
import os
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from dataset.session import MouseChoice, MouseCue, MouseSession
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

R_MIN = 0.0
R_MAX = 1.0




def gen_dataset_trial(t, ms, savefile, result_file, s_cue, e_cue):
    trial = ms[t]
    #print(len(trial.ts))
    start_frame = trial.get_frame(s_cue)
    end_frame = trial.get_frame(e_cue)
    pr = np.corrcoef(trial.ts[start_frame:end_frame, :].transpose())
    pr = pr.clip(min=R_MIN, max=R_MAX)
    for row in range(len(pr)):
        for entry in range(row):
            if True:
                pr[row][entry] = .0001
    pr[np.isnan(pr)] = 0
    pr = torch.Tensor(pr)
    dataset = data.TensorDataset(pr)
    loader = DataLoader(dataset)
    # return pr
    return loader

class GCN(pl.LightningModule):
    def __init__(self, adj_mat_dim, num_features):
        super().__init__()
        self.adj_mat = torch.Tensor(adj_mat_dim, adj_mat_dim) # Should we pass in a normal Tensor or a DataLoader?
        self.features = torch.rand(len(self.adj_mat), num_features) # What range should the features initalize in?        
        # Should we just initialize self.adj_mat and self.features to be torch.nn.Linear objects of the proper shape
        # And construct the other matrices listed below in the forward pass?
        self.lambda_mat = torch.diag(torch.ones(len(self.adj_mat), requires_grad = True))
        self.adj_tilde = torch.add(self.adj_mat, self.lambda_mat)
        self.degree_mat_inv = torch.diag(self.adj_tilde.clone().sum(1).pow(-1/2))
    
    def forward(self):
        return torch.matmul(
                    torch.matmul(
                        torch.matmul(
                            self.degree_mat_inv, 
                            self.adj_mat), 
                        self.degree_mat_inv), 
                    self.features)


def main():
    dirs = ['./data/RH730_170510']

    res_dir = "./results"
    for base_path in dirs:
        m_tag = re.search(r'(\w+)_', base_path).group(1)
        result_data_file = "./results/correlation_matrices" + m_tag + ".pk"
        # prep results dir
        save_dir = os.path.join(res_dir, m_tag)
        os.makedirs(save_dir, exist_ok=True)


    for t in tqdm(range(10,11)):
        ms = MouseSession(base_path=base_path)
        save_path = os.path.join(save_dir,
                                "fixed_{m_tag}_{trial_no}.png".format(m_tag=m_tag,
                                                                        trial_no=t))
        data = gen_dataset_trial(t, ms, savefile=save_path, result_file=result_data_file, s_cue=MouseCue.ANS, e_cue=MouseCue.REWARD)

        # gcn = GCN(data, len(data))
        gcn = GCN(
        print(gcn)

if __name__ == '__main__':
    main()

