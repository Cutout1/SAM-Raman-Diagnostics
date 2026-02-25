import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigurableCNN1D(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 input_length=100,
                 cnn_configs=[
                     {"filters": 60, "kernel_size": 60, "stride": 1},
                     {"filters": 70, "kernel_size": 11, "stride": 4},
                     {"filters": 93, "kernel_size": 45, "stride": 1}
                 ],
                 pool_configs=[
                     {"kernel_size": 5, "stride": 5},
                     {"kernel_size": 4, "stride": 4},
                     {"kernel_size": 2, "stride": 2}
                 ],
                 fc_units=[360, 224, 122],
                 dropout_rate=[0.1, 0.5, 0],
                 num_classes=3,
                 activation='relu'):
        super(ConfigurableCNN1D, self).__init__()
        
        # Convolution + BatchNorm + Pooling layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = input_channels
        seq_len = input_length
        self.activation = activation
        
        for i, cfg in enumerate(cnn_configs):
            print("seq_len: " + str(seq_len))
            conv = nn.Conv1d(in_channels=in_channels,
                             out_channels=cfg["filters"],
                             kernel_size=cfg["kernel_size"],
                             stride=cfg["stride"],
                             padding=cfg["kernel_size"])
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(cfg["filters"]))
            
            pool_cfg = pool_configs[i]
            pool = nn.MaxPool1d(kernel_size=pool_cfg["kernel_size"],
                                stride=pool_cfg["stride"])
            self.pools.append(pool)
            
            # update sequence length after conv + pool
            seq_len = self._compute_output_length(seq_len,
                                                  cfg["kernel_size"],
                                                  cfg["stride"],
                                                  cfg["kernel_size"])
            seq_len = self._compute_output_length(seq_len,
                                                  pool_cfg["kernel_size"],
                                                  pool_cfg["stride"],
                                                  )
            in_channels = cfg["filters"]
        print("seq_len: " + str(seq_len))
        # Flatten dimension after convs/pools
        flatten_dim = in_channels * seq_len
        print("flatten_dim: " + str(flatten_dim))
        # Fully connected layers
        layers = []
        prev_units = flatten_dim
        unit_index = 0
        for units in fc_units:
            layers.append(nn.Linear(prev_units, units))
            if(activation=='gelu'):
                layers.append(nn.GELU())
            elif(activation=='relu'):
                layers.append(nn.ReLU())
            elif(activation=='selu'):
                layers.append(nn.SELU())
            layers.append(nn.Dropout(dropout_rate[unit_index]))
            prev_units = units
            unit_index += 1
        layers.append(nn.Linear(prev_units, num_classes))
        
        self.fc_seq = nn.Sequential(*layers)

    def _compute_output_length(self, L_in, kernel_size, stride, padding=0, dilation=1):
        """
        Compute output length of 1D conv/pool layer
        Formula: L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        """
        return (L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = conv(x)
            x = bn(x)
            if(self.activation=='gelu'):
                x = F.gelu(x)
            elif(self.activation=='relu'):
                x = F.relu(x)
            elif(self.activation=='selu'):
                x = F.selu(x)
            x = pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc_seq(x)
        return x