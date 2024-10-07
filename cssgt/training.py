import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_loss(postive, negative, margin=0.0):
   cos = nn.CosineEmbeddingLoss()
   postive_1 = postive.unsqueeze(0)
   negative_1 = negative.unsqueeze(0)
   output = cos(postive_1, negative_1, target=torch.ones_like(postive))
   return output

#@measure_energy
def model_train(model, optimizer, train_data, feature_matrix):
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    z1s, z2s = model(feature_matrix, train_data.edge_index, train_data.edge_attr)
    for z1, z2 in zip(z1s, z2s):
      loss = cos_loss(z1, z2)
      loss.backward()
      loss_total += loss.item()
    optimizer.step()
    return loss_total

