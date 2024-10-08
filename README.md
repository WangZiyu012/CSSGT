# 1. Run the code

Simply run the 6 files to reproduce the results! The brief description of files are given below:

1. **run.py**: quickly see the performance of CSSGT without saving.
2. **main.py**: quickly reproduce the accuracy curve of CSSGT and save.
3. **ablation_gnn.py**: reproduce the ablation study on the hidden dimension of graph convolution layers.
4. **ablation_migs.py**: reproduce the ablation study on the number of partitions.
5. **ablation_sn.py**: reproduce the ablation study on the type of SNs (IF, LIF, and PLIF).
6. **ablation_threshold.py**: reproduce the ablation study on the threshold of SNs.

# 2. Main Requirements

python == 3.8

torch == 2.0.0

torch_geometric ==2.3.0

CUDA 11.8

# 3. CSSGT Architecture

1. Overall architecture.
   ![Arc1](figs/framework.jpg)

2. Detailed architecture.
   ![Arc2](figs/CSSGT.jpeg)

3. Architecture comparison between Spike-Driven Graph Attention and other attention mechanisms.
   ![Arc3](figs/SDGA.jpg)
