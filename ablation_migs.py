import numpy as np
import torch
import warnings
import pandas as pd

from torch_geometric import seed_everything

from cssgt.evaluate import test
from cssgt.dataset import get_dataset
from cssgt.training import model_train

from cssgt.cssgt import CSSGT
from cssgt.spliting import split_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "./data"
seed = 42
lr = 0.01
Epoch = 50
warnings.filterwarnings("ignore", category=UserWarning)

# load data
split_range = [1,10,20,40,60,80,100]
data_name = "Cora"
result_vec = np.zeros(len(split_range)) 

for i, num_split in enumerate(split_range):

   data = get_dataset(root=root,dataset=data_name)
   data = data.to(device)
   seed_everything(seed)

   feature_subsets, group_sizes = split_features(data, T=num_split)



   # iterate snn and sa threshold
   model = CSSGT(in_channels=group_sizes, threshold=0.5, if_multi_output= False).to(device)

   print(f"Training model with split={num_split}")
   model_acc_curve = []
   model_loss_curve = []

   optimizer = torch.optim.AdamW(params=model.parameters(),
                        lr=lr)

   for epoch in range(1, Epoch + 1):

      loss = model_train(model, optimizer, train_data=data, feature_matrix=feature_subsets)
      model.eval()
      with torch.no_grad():
         embeds = model.enc(feature_subsets, data.edge_index, data.edge_attr)
         embeds = torch.cat(embeds, dim=-1)
      _, tests_output = test(embeds, data, data.num_classes)

      print(f"Epoch: {int(epoch)}")
      print(f"Mean: {round(np.mean(tests_output)*100,2)}%")

      epoch_median = np.median(tests_output)
      epoch_best = np.max(tests_output)
      
      # record the loss and test result
      model_loss_curve.append(loss)
      model_acc_curve.append(epoch_median)

   

   model_performance = np.max(model_acc_curve)
   print(f"Model result: {round(model_performance *100,2)}%")
   result_vec[i] = model_performance

   # save the loss
   df_loss = pd.DataFrame(model_loss_curve)
   df_loss.to_csv(f'results/split_{data_name}/loss/split_loss_{num_split}_{data_name}.csv', index=False, header=False)

   # save the model_acc_curve
   df_acc = pd.DataFrame(model_acc_curve)
   df_acc.to_csv(f'results/split_{data_name}/accuracy/split_acc_{num_split}_{data_name}.csv', index=False, header=False)

   # save the result matrix
df_matrix = pd.DataFrame(result_vec)
df_matrix.to_csv(f'results/split_{data_name}/split_all_models_matrix_{data_name}.csv', index=False, header=False)
