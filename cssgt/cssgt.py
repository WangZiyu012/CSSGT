import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge, to_torch_sparse_tensor
from cssgt.spiking import creat_snn_layer



class CSSGT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels: int = 64,
        out_channels: int = 64,
        threshold: float = 0.7,
        snn_type: str = "PLIF",
        reset: str = "zero",
        if_reset: bool = True,
        sgpe: str = "yes",
        num_attention_heads=1,
        if_multi_output: bool = False,
    ):
        super().__init__()
        self.threshold = threshold
        self.snn_type = snn_type
        self.reset = reset
        self.if_reset = if_reset
        self.sgpe = sgpe
        self.num_attention_heads = num_attention_heads
        self.hidden_channels = hidden_channels 
        self.out_channels = out_channels
        self.if_multi_output = if_multi_output  

        self.q_lins = GCNConv(self.hidden_channels, self.hidden_channels)
        self.k_lins = GCNConv(self.hidden_channels, self.hidden_channels)   
        self.conv = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin = torch.nn.Linear(self.hidden_channels, self.out_channels, bias=False)
        self.v_lins = GCNConv(self.hidden_channels, self.hidden_channels)
        self.pos_conv = GCNConv(self.hidden_channels, self.hidden_channels)
        self.att_conv = GCNConv(self.hidden_channels, self.hidden_channels)

        self.att_bn = BatchNorm1d(self.hidden_channels)
        self.att_bn_2 = BatchNorm1d(self.hidden_channels)
        self.pos_bn = BatchNorm1d(self.hidden_channels)
        self.pos_bn_2 = BatchNorm1d(self.hidden_channels)
        self.bn_0 = BatchNorm1d(self.hidden_channels)
        self.bn_1 = BatchNorm1d(self.hidden_channels)
        self.bn_2 = BatchNorm1d(self.hidden_channels)
        self.snn_bn = BatchNorm1d(self.hidden_channels)
        self.q_bn = BatchNorm1d(self.hidden_channels)
        self.k_bn = BatchNorm1d(self.hidden_channels)
        self.v_bn = BatchNorm1d(self.hidden_channels)

        
        

        self.convs = torch.nn.ModuleList()
        for channel in in_channels:
            self.convs.append(GCNConv(channel, self.hidden_channels))

        # spiking layers
        self.q_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.k_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.pos_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.conv_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.conv_lif_2 = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.attn_lif = creat_snn_layer(v_threshold=0.2, snn=self.snn_type)
        self.v_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        self.final_lif = creat_snn_layer(v_threshold=self.threshold, snn=self.snn_type)
        

    def enc(self, input, edge_index, edge_weight=None):
        xs = []

        if self.if_multi_output:
            conv_fr = []
            pos_fr = []
            q_fr = []
            k_fr = []
            v_fr = []
            att_fr = []




        edge_index = to_torch_sparse_tensor(edge_index, size=len(input[0]))

        

        # iniitialization
        

        for i, x in enumerate(input):
                      
            # GCN layers   
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bn_1(x)

            x = self.conv_lif(x)
            if self.if_multi_output:
                conv_fr.append(x)
            
            x = self.conv(x, edge_index, edge_weight)
            x = self.bn_2(x)
            


            # position embedding
            if self.sgpe == "yes":
                x_pos_spike = self.pos_lif(x)
                x_pos = self.pos_conv(x_pos_spike, edge_index, edge_weight)
                x_pos = self.pos_bn(x_pos)
                x = x_pos + x
                x = self.pos_bn_2(x)

                if self.if_multi_output:
                    pos_fr.append(x_pos_spike)
            

            if self.num_attention_heads != 0:
                # attention layers
                x_att = self.conv_lif(x)
                q_linear_out = self.q_lins(x_att, edge_index, edge_weight)
                q_linear_out = self.q_bn(q_linear_out).contiguous()
                q = self.q_lif(q_linear_out)
                

                k_linear_out = self.k_lins(x_att, edge_index, edge_weight)
                k_linear_out = self.k_bn(k_linear_out).contiguous()
                k = self.k_lif(k_linear_out)

                v_linear_out = self.v_lins(x_att, edge_index, edge_weight)
                v_linear_out = self.v_bn(v_linear_out).contiguous()
                v = self.v_lif(v_linear_out)

                a,b = q.shape
                
                if self.if_multi_output:
                    q_fr.append(q)
                    k_fr.append(k)
                    v_fr.append(v)

                if self.num_attention_heads > 1:
                    q = q.reshape(a, self.num_attention_heads, b// self.num_attention_heads)
                    k = k.reshape(a, self.num_attention_heads, b// self.num_attention_heads)
                    v = v.reshape(a, self.num_attention_heads, b// self.num_attention_heads)
                            
                qk = q.mul(k)

                if self.num_attention_heads > 1:
                    qk = qk.mean(axis=2)
                else:
                    qk = qk.mean(axis=1)

                # kv spike
                qk = self.attn_lif(qk)


                if self.num_attention_heads > 1:
                    qk = qk.unsqueeze(2)
                    qk = qk.repeat(1, 1, b//self.num_attention_heads)

                else:
                    qk = qk.repeat(self.hidden_channels,1).T

                # qkv (Hadamard product)
                att = qk.mul(v)

                if self.num_attention_heads > 1:
                    att = att.reshape(a, b)
                    
                if self.if_multi_output:
                    att_fr.append(att)                   
                    

                att = self.att_conv(att, edge_index, edge_weight)
                att = self.att_bn(att)
                x = x + 0.125*att 
                x = self.att_bn_2(x)




            xs.append(x)        
              
               
            
                 
        
        #reset the neurons
        if self.if_reset == True:      
            self.q_lif.reset(self.reset)
            self.k_lif.reset(self.reset)
            self.v_lif.reset(self.reset)
            self.attn_lif.reset(self.reset)
            self.pos_lif.reset(self.reset)


        if self.if_multi_output:
            return xs, pos_fr, conv_fr, q_fr, k_fr, v_fr, att_fr 

        else:
            return xs
    

    def forward(self, x, edge_index, edge_weight=None):
        
        edge_index2, mask2 = dropout_edge(edge_index, p=0.2)

        if edge_weight is not None:
            edge_weight2 = edge_weight[mask2]
        else:
            edge_weight2 = None

        x2 = x

        if self.if_multi_output:
            s1,_,_,_,_,_,_ = self.enc(x, edge_index, edge_weight)
            s2,_,_,_,_,_,_ = self.enc(x2, edge_index2, edge_weight2)

        else:
            s1 = self.enc(x, edge_index, edge_weight)
            s2 = self.enc(x2, edge_index2, edge_weight2)

        z1 = self.dec(s1)
        z2 = self.dec(s2)
        
        return z1, z2
    
    def dec(self, input):
        output = []
        for x in input:
            output.append(self.lin(x).sum(1))
        return output