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
data_list = ["Cora"]

for data_name in data_list:

   data = get_dataset(root=root,dataset=data_name)
   data = data.to(device)
   seed_everything(seed)

   feature_subsets, group_sizes = split_features(data, T=30)

   # define the range of snn and sa threshold
   threshold_range = [0.1, 0.3, 0.5, 0.7, 0.9]


   result_vec = np.zeros(len(threshold_range))   



   # iterate snn and sa threshold
   for i, a1 in enumerate(threshold_range):
      model = CSSGT(in_channels=group_sizes ,threshold=a1, if_multi_output= True).to(device)

      print(f"Training model with threshold={a1}")
      model_acc_curve = []
      model_loss_curve = []
      FR_pos = []
      FR_conv = []
      FR_q = []
      FR_k = []
      FR_v = []
      FR_att = []

      optimizer = torch.optim.AdamW(params=model.parameters(),
                           lr=lr)

      for epoch in range(1, Epoch + 1):

         loss = model_train(model, optimizer, train_data=data, feature_matrix=feature_subsets)
         model.eval()
         with torch.no_grad():
            embeds, conv, pos, q, k, v, att  = model.enc(feature_subsets, data.edge_index, data.edge_attr)
            embeds = torch.cat(embeds, dim=-1)
            pos = torch.cat(pos, dim=-1)
            conv = torch.cat(conv, dim=-1)
            q = torch.cat(q, dim=-1)
            k = torch.cat(k, dim=-1)
            v = torch.cat(v, dim=-1)
            att = torch.cat(att, dim=-1)
         _, tests_output = test(embeds, data, data.num_classes)

         print(f"Epoch: {int(epoch)}")
         print(f"Round_mean: {round(np.mean(tests_output)*100,2)}%")
         print(f"FR_pos: {round(pos.mean().item()*100,2)}%")
         print(f"FR_conv: {round(conv.mean().item()*100,2)}%")
         print(f"FR_q: {round(q.mean().item()*100,2)}%")
         print(f"FR_k: {round(k.mean().item()*100,2)}%")
         print(f"FR_v: {round(v.mean().item()*100,2)}%")
         print(f"FR_att: {round(att.mean().item()*100,2)}%")

         epoch_median = np.median(tests_output)
         epoch_best = np.max(tests_output)
         
         # record the loss and test result
         model_loss_curve.append(loss)
         model_acc_curve.append(epoch_median)

         # record the firing rates
         FR_pos.append(round(pos.mean().item(), 4))
         FR_conv.append(round(conv.mean().item(), 4))
         FR_q.append(round(q.mean().item(), 4))
         FR_k.append(round(k.mean().item(), 4))
         FR_v.append(round(v.mean().item(), 4))
         FR_att.append(round(att.mean().item(), 4))

      

      model_performance = np.max(model_acc_curve)
      print(f"Model result: {round(model_performance *100,2)}%")
      result_vec[i] = model_performance.item()

      # save the loss
      df_loss = pd.DataFrame(model_loss_curve)
      df_loss.to_csv(f'results/ablation_threshold_{data_name}/loss/threshold_loss_{data_name}_{a1}.csv', index=False, header=False)

      # save the model_acc_curve
      df_acc = pd.DataFrame(model_acc_curve)
      df_acc.to_csv(f'results/ablation_threshold_{data_name}/accuracy/threshold_acc_{data_name}_{a1}.csv', index=False, header=False)

      # save the firing rate of the encoder, Q and K
      df_FR_enc = pd.DataFrame(FR_pos)
      df_FR_enc.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_pos/threshold_FR_pos_{data_name}_{a1}.csv', index=False, header=False)

      df_FR_conv = pd.DataFrame(FR_conv)
      df_FR_conv.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_conv/threshold_FR_conv_{data_name}_{a1}.csv', index=False, header=False)

      df_FR_q = pd.DataFrame(FR_q)
      df_FR_q.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_q/threshold_FR_q_{data_name}_{a1}.csv', index=False, header=False)

      df_FR_k = pd.DataFrame(FR_k)
      df_FR_k.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_k/threshold_FR_k_{data_name}_{a1}.csv', index=False, header=False)

      df_FR_v = pd.DataFrame(FR_v)
      df_FR_v.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_v/threshold_FR_v_{data_name}_{a1}.csv', index=False, header=False)

      df_FR_att = pd.DataFrame(FR_att)
      df_FR_att.to_csv(f'results/ablation_threshold_{data_name}/firing_rate_att/threshold_FR_att_{data_name}_{a1}.csv', index=False, header=False)

   # save the result matrix
   df_matrix = pd.DataFrame(result_vec)
   df_matrix.to_csv(f'results/ablation_threshold_{data_name}/threshold_all_models_matrix_{data_name}.csv', index=False, header=False)
