# CSSGT Architecture

Overall architecture
![Arc1](figs/framework.jpg)

Detailed architecture
![Arc2](figs/CSSGT.jpeg)

Architecture comparison between Spike-Driven Graph Attention and other attention mechanisms
![Arc3](figs/SDGA.jpg)

# Run the code

This is a guidance of running CSSGT.

Simply run the 6 files to reproduce the results!

1. **ablation_gnn.py**: reproduce the ablation study on the hidden dimension of graph convolution layers.
2. **ablation_migs.py**: reproduce the ablation study on the number of partitions.
3. **ablation_sn.py**: reproduce the ablation study on the type of SNs (IF, LIF, and PLIF).
4. **ablation_threshold.py**: reproduce the ablation study on the threshold of SNs.
5. **main.py**: quickly reproduce the accuracy curve of CSSGT and save.
6. **run.py**: quickly see the performance of CSSGT without saving.

# Main Requirements

python == 3.8

torch == 2.0.0

torch_geometric ==2.3.0

CUDA 11.8
