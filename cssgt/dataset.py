import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikipediaNetwork, Actor, Reddit, Flickr, Yelp, WebKB

def get_dataset(root, dataset, num_val=0.1, num_test=0.8):

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(name=dataset, root=root, split="full")
        data = dataset[0]

    elif dataset in {"Chameleon", "Squirrel"}:
        dataset = WikipediaNetwork(root, dataset, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)

    elif dataset in {"Photo", "Computers"}:
        dataset = Amazon(root, dataset, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)

    elif dataset in {"Actor"}:
        dataset = Actor(root, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)

    elif dataset in {"CS", "Physics"}:
        dataset = Coauthor(root, dataset, transform=T.ToUndirected())
        data = dataset[0]
        data = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(data)
        
    else:
        raise ValueError(dataset)
    data.num_classes = dataset.num_classes
    return data
