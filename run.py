import numpy as np
import torch
import warnings

from torch_geometric import seed_everything
from torch_geometric.logging import log

from cssgt.evaluate import test
from cssgt.dataset import get_dataset
from cssgt.training import model_train
from cssgt.cssgt import CSSGT
from cssgt.spliting import split_features


# This is a quick start to see how CSGQT performs on a series of datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "./data"
seed = 42
lr = 0.01
Epoch = 50
warnings.filterwarnings("ignore", category=UserWarning)


# load data
# Note: you can change the dataset names to any of the following datasets, for this example we are using Cora, Citeseer and Squirrel.
# Options: "Chameleon", "Cora", "Citeseer", "Squirrel", "Photo", "Actor", "Computers", "Pubmed"

data_name = "Cora"

data = get_dataset(root=root,dataset=data_name)
data = data.to(device)
seed_everything(seed)

feature_subsets, group_sizes = split_features(data, T=30)

print(f"Sizes of Feature groups: {group_sizes}")



# Model initialization
# You can add more models to the list to compare them, for this example we use qk-pca

model = CSSGT(in_channels=group_sizes, if_multi_output=False).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Training loop

print(f"Training on dataset {data_name}")
print(f"Total number of parameters: {count_parameters(model)}")
best = 0
tests = []

optimizer = torch.optim.AdamW(params=model.parameters(),
                        lr=lr)
for epoch in range(1, Epoch+1):
    loss = model_train(model, optimizer, train_data= data,feature_matrix=feature_subsets)
    model.eval()


    with torch.no_grad():
        embeds = model.enc(feature_subsets, data.edge_index, data.edge_attr)
        embeds = torch.cat(embeds, dim=-1)
        print("=" * 42)
        print(f"Epoch: {int(epoch)}")
        print("." * 42)
    val_output, tests_output = test(embeds, data, data.num_classes)
    test_median = np.median(tests_output)
    test_best = np.max(tests_output)
    val_median = np.mean(val_output)
    val_best = np.max(val_output)
    tests.append(test_median)

    if test_best > best:
        best = test_best
    
    
    
    log(Loss=loss,best=best)
    log(round_best=test_best, round_median = test_median)

print("%" * 100)
log(FINAL_BEST=best)
log(Final_MEdiAN=np.max(tests))
print("%" * 100)
      

