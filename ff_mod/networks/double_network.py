import torch
from torch.optim import Adam, SGD

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet, AbstractForwardForwardLayer

from ff_mod.overlay import Overlay

class DoubleNetwork(AbstractForwardForwardNet):

    def __init__(
            self,
            overlay_function : Overlay,
            dims,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 20,
            first_prediction_layer = 0,
            residual_connections = False,
            normalize_activity = False,
        ):
        
        super().__init__(overlay_function, first_prediction_layer, residual_connections)
        
        self.internal_epoch = internal_epoch
        self.normalize_activity = normalize_activity
        
        for d in range(len(dims) - 1):
            if residual_connections and d > 0:
                layer = DobuleLayer(dims[d] + dims[0], dims[d + 1], loss_function, learning_rate, internal_epoch)
            else:
                layer = DobuleLayer(dims[d], dims[d + 1], loss_function, learning_rate, internal_epoch)
            
            self.layers.append(layer)

        self.layers[0].is_input_layer = True
        
    def adjust_data(self, data):
        return data
    
    def reset_linear_weights(self):
        for layer in self.layers:
            layer.set_linear_weights(None)
            
    def set_linear_factor(self, factor):
        for layer in self.layers:
            layer.linear_factor = factor

class DobuleLayer(AbstractForwardForwardLayer):
    def __init__(
            self,
            in_features,
            out_features,
            loss_function,
            learning_rate = 0.001,
            internal_epoch = 10,
            normalize_activity = False,
            **kwargs
        ):
        super().__init__(in_features, out_features, loss_function = loss_function, **kwargs)
        
        self.is_input_layer = False
        
        #self.relu = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_features)
        
        self.opt = Adam(self.parameters(), lr=learning_rate)
        #self.opt = SGD(self.parameters(), lr=learning_rate)
        self.num_epochs = internal_epoch
        
        self.normalize_activity = normalize_activity
        
        self.linear_weights = None
        self.linear_factor = 1

    def set_linear_weights(self, linear_weights):
        # We recieve a count of the activity of all neurons
        # We must apply a functin
        if linear_weights is None:
            self.linear_weights = None
            return
        
        normalized_weights = linear_weights / linear_weights.sum()
        
        self.linear_weights = torch.exp(-1000 * self.linear_factor * normalized_weights.unsqueeze(0))

    def get_goodness(self, x):
        # We have defined the latent as [Pos, Neg]
        # We define the goodness as the ratio between the positive and the negative
        
        mid_point = x.size(1) // 2
        # Normalize the vector x for each row
        #x = x / (x.pow(2).mean(1).sqrt().unsqueeze(1) + 1e-10)
        
        goodness =  (x[:, :mid_point].pow(2).mean(1)+1e-9)/(2 * x.pow(2).mean(1)+2e-9)
        
        l = x[:, :mid_point].pow(2).mean(1)
        r = x[:, mid_point:].pow(2).mean(1)
        
        
        goodness = torch.clamp(goodness, 0+0.0001, 1-0.0001)
        
        return goodness
    
    def forward(self, x):
        
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)
        
        #if not self.is_input_layer:
        #    x = self.bn(x)

        if not self.is_input_layer and self.normalize_activity:
            # Since it is a double network, normalize both parts independently
            
            mid_size = x.size(1) // 2
            
            x[:mid_size] = x[:mid_size] / (x[:mid_size].pow(2).mean(1).sqrt().unsqueeze(1) + 1e-10)
            x[mid_size:] = x[mid_size:] / (x[mid_size:].pow(2).mean(1).sqrt().unsqueeze(1) + 1e-10)
        
        result =  self.relu(torch.mm(x, self.weight.T) + bias_term)
        
        mid_size = result.size(1) // 2
        
        val_ex, ind_ex = torch.topk(result[:,:mid_size], 15, dim=1)
        val_in, ind_in = torch.topk(result[:,mid_size:], 15, dim=1)
        # Set all but the top k to zero
        
        # Create a new vector with those values 
        result = torch.zeros_like(result)
        result = result + torch.zeros_like(result).scatter(1, ind_ex, val_ex)
        result = result + torch.zeros_like(result).scatter(1, mid_size + ind_in, val_in)
        
        
        if self.linear_weights is None:
            return result
        
        return result * self.linear_weights
