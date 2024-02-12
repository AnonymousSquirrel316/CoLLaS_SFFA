import torch
import torch.nn as nn
import torch.nn.functional as F

from ff_mod.overlay import Overlay

from abc import ABC, abstractmethod
from typing import List

from avalanche.models.dynamic_modules import MultiTaskModule, DynamicModule

from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.benchmarks.scenarios import CLExperience


class FF_Network(nn.Module):
    """
    Abstract class representing a FeedForward Forward-Forward Neural Network.
    
    Ref:
        - The Forward-Forward Algorithm: Some Preliminary Investigations - G. Hinton (https://arxiv.org/pdf/2212.13345.pdf)
    """
    
    def __init__(
            self,
            residual_connections = False,
            normalize_activity = False):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        
        self.overlay_function = None
        self.total_classes = 10
        
        self.residual_connections = residual_connections
        self.normalize_activity = normalize_activity
        
    def add_layer(self, layer : nn.Module):
        self.layers.append(layer)
    
    def goodness(self, x):
        mid = x.shape[1] // 2
                
        goodness =  (x[:, :mid].pow(2).sum(1) + 1e-9)/(x.pow(2).sum(1) + 2e-9)
             
        # Done to avoid divisions by zero or problems in loss
        goodness = torch.clamp(goodness, 0.0001, 0.9999)
        
        return goodness
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Returns the average goodness, which is a probability.
        
        x in training will be (x,y) pair
        
        Note: It is assumed that x includes the label in it. It is also assumed that the data is adjusted.
        """
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        if self.training:
            x_init = x.clone().detach()
            
            goodnesses = torch.zeros(x.shape[0], self.total_classes, len(self.layers), requires_grad=True).to(x.device)
            
            for label in range(self.total_classes):
                input_vec = self.overlay_function(x_init.clone().detach(), torch.full((x.shape[0],), label, dtype=torch.long).to(x.device))
                x_base = input_vec.clone().detach()
                
                
                for i, layer in enumerate(self.layers):
                    
                    if self.residual_connections and i > 0:
                        input_vec = torch.cat([input_vec.clone().detach(), x_base.clone().detach()], 1)
                    else:
                        input_vec = input_vec.clone().detach()
                        
                    input_vec = layer(input_vec)
                    
                    goodnesses[:, label, i] = self.goodness(input_vec)
            
            # MULTIPLY on the last dimension to get the average goodness
            goodnesses = goodnesses.prod(2)
            
            return goodnesses
        
        else:
            # TODO Tecnicamente, si es no training se puede simplemente hacer el mean, pero lo de arriba es igual...
            goodness = torch.zeros(x.shape[0], self.total_classes).to(x.device)
            
            x_init = x.clone().detach()
            
            for label in range(self.total_classes):
                input_vec = self.overlay_function(x_init.clone().detach(), torch.full((x.shape[0],), label, dtype=torch.long).to(x.device))
                x_base = input_vec.clone().detach()
                
                cur_goodness = torch.zeros(x.shape[0]).to(x.device)
                counter = 0
                
                for i, layer in enumerate(self.layers):
                    if self.residual_connections and i > 0:
                        input_vec = torch.cat([input_vec.clone().detach(), x_base.clone().detach()], 1)
                    else:
                        input_vec = input_vec.clone().detach()
                    
                    input_vec = layer(input_vec)
                    
                    cur_goodness += self.goodness(input_vec)
                    counter += 1
                
                goodness[:, label] = cur_goodness
                goodness[:, label] /= counter
            
            return goodness

class FF_Layer(nn.Linear):
    """
    Class representing a layer in a Feed-Forward Neural Network.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            device="cuda:0",
            dtype=None
        ):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.activation = nn.Sigmoid()
        
        self.is_input = False
        self.normalize_activity = False


    def forward(self, x, mask = None):
        bias_term = 0
        if self.bias is not None:
            bias_term = self.bias.unsqueeze(0)
            
        mid = self.out_features // 2
        
        if not self.is_input and self.normalize_activity:
            x[:mid] = x[:mid] / (x[:mid].norm(1) + 1e-10)
            x[mid:] = x[mid:] / (x[mid:].norm(1) + 1e-10)
        
        result =  self.activation(torch.mm(x, self.weight.T) + bias_term)
        
        if mask is not None:
            result = result * mask
        
        # Set all but the top k to zero
        val_ex, ind_ex = torch.topk(result[:,:mid], 15, dim=1)
        val_in, ind_in = torch.topk(result[:,mid:], 15, dim=1)
        
        # Create a new vector with those values 
        result = torch.zeros_like(result)
        result = result + torch.zeros_like(result).scatter(1, ind_ex, val_ex)
        result = result + torch.zeros_like(result).scatter(1, mid + ind_in, val_in)
        
        return result

class FF_Network_Incremental(DynamicModule):
    
    def __init__(
        self,
        initial_out_features=2,
        residual_connections = False,
        normalize_activity = False,
        masking = False
    ):
        super().__init__()
                
        self.layers = torch.nn.ModuleList()
        
        self.cur_classes = initial_out_features
        
        self.residual_connections = residual_connections
        self.normalize_activity = normalize_activity
        
        self.cur_nclasses = initial_out_features
        self.cur_classes = []
        
        self.mask_per_layer = {}
        self.mask_per_label_layer = {}
        self.neg_mask_per_layer = {}
        self.neg_mask_per_label_layer = {}
        self.masking = masking
    
    def add_layer(self, layer : nn.Module):
        self.layers.append(layer)
        
        n_layers = len(self.layers)
        
        self.mask_per_layer[n_layers-1] = torch.full((self.layers[-1].out_features,), 1.0).to('cuda:0')
        self.neg_mask_per_layer[n_layers-1] = torch.full((self.layers[-1].out_features,), 1.0).to('cuda:0')
        self.mask_per_label_layer[n_layers - 1] = {}
        self.neg_mask_per_label_layer[n_layers - 1] = {}

    def goodness(self, x):
        mid = x.shape[1] // 2
                
        goodness =  (x[:, :mid].pow(2).sum(1) + 1e-9)/(x.pow(2).sum(1) + 2e-9)
             
        # Done to avoid divisions by zero or problems in loss
        goodness = torch.clamp(goodness, 0.0001, 0.9999)
        
        return goodness

    def get_mask(self, layer, label = None):    
        if label is not None:
            if label not in self.mask_per_label_layer[layer]:
                self.mask_per_label_layer[layer][label] = torch.full((self.layers[layer].out_features,), 1.0).to('cuda:0')
                self.neg_mask_per_label_layer[layer][label] = torch.full((self.layers[layer].out_features,), 1.0).to('cuda:0')
            
            return self.mask_per_label_layer[layer][label] * self.neg_mask_per_label_layer[layer][label]
        else:    
            return self.mask_per_layer[layer] * self.neg_mask_per_layer[layer]

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        
        old_nclasses = self.cur_nclasses
        curr_classes = experience.classes_in_this_experience
        
        new_nclasses = len(curr_classes) - old_nclasses
        
        classes = set(curr_classes).union(set(self.cur_classes))
        
        self.cur_nclasses = len(classes)
        self.cur_classes = list(classes)
        self.cur_classes.sort()
                    
        print("New classes: ", self.cur_classes)

        # Currently we will always start with number of 
        #  pattern greater than total number of classes

    def get_latent(self, x, label, layer_out, use_masking = True):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        x_init = x.clone().detach()
        labels = torch.full((x.shape[0],), label, dtype=torch.long).to(x.device)
        
        input_vec = self.overlay_function(x_init, labels)
            
        x_base = input_vec.clone().detach()
        
        for i, layer in enumerate(self.layers):
            
            if self.residual_connections and i > 0:
                input_vec = torch.cat([input_vec.clone().detach(), x_base.clone().detach()], 1)
            else:
                input_vec = input_vec.clone().detach()
            
            if self.masking and use_masking:
                if self.training:
                    input_vec = layer(input_vec, self.get_mask(i))
                else:
                    input_vec = layer(input_vec, self.get_mask(i, label))
            else:
                input_vec = layer(input_vec)
                
            if i == layer_out:
                return input_vec

    def compute_mask(self, n_samples : int, experience: CLExperience):
        """Computes the mask for the current experience.

        # Currently assuming that tasks have 2 classes, and that they appear in order

        :param experience: data from the current experience.
        :return:
        """
        
        total_size = len(experience.dataset)
        
        cur_classes = experience.classes_in_this_experience
        
        print("Computing mask for classes: ", cur_classes)
        print(f"Is currently {self.training}")
        
        # TODO Fix after
        batch = {}
        for cur_class in cur_classes:
            batch[cur_class] = torch.zeros(n_samples, 784).to('cuda:0')
            
        for i in range(10):
            batch[cur_classes[0]][i] = experience.dataset[i][0].reshape(1, 784)
        print(f"Expected lable {0} but got {experience.dataset[i][1]}")
        
        for i in range(10):
            batch[cur_classes[1]][i] = experience.dataset[total_size-i-1][0].reshape(1, 784)
        print(f"Expected lable {1} but got {experience.dataset[total_size-1][1]}")
                
        # Positive
        for cur_class in cur_classes:
            for layer in range(len(self.layers)):
                latent = self.get_latent(batch[cur_class], cur_class, layer, use_masking = True)
                
                # latent (batch_size, n_features)
                latent = latent.sum(0)
                # Add random noise to avoid problems with zero values
                latent += torch.randn_like(latent) * 1e-7
                
                mid_size = latent.shape[0] // 2
                
                # Get a mask for only the top k values
                val, ind = torch.topk(latent[:mid_size, ], 70, dim=0)
                
                # Create a new vector with those values
                mask = 1 - torch.zeros_like(latent).scatter(0, ind, 1)
                
                self.mask_per_layer[layer] *= mask
                self.mask_per_label_layer[layer][cur_class] = 1 - mask
                self.mask_per_label_layer[layer][cur_class][mid_size:] = 1
        
        # Negative
        for j, cur_class in enumerate(cur_classes):
            for layer in range(len(self.layers)):
                # Latent of images from class with other label
                latent = self.get_latent(batch[cur_class], cur_classes[1-j], layer, use_masking = True)
                
                # latent (batch_size, n_features)
                latent = latent.sum(0)
                # Add random noise to avoid problems with zero values
                latent += torch.randn_like(latent) * 1e-7
                
                mid_size = latent.shape[0] // 2
                
                # Get a mask for only the top k values
                val, ind = torch.topk(latent[mid_size:, ], 70, dim=0)
                
                # Create a new vector with those values
                mask = 1 - torch.zeros_like(latent).scatter(0, mid_size + ind, 1)
                
                self.neg_mask_per_layer[layer] *= mask
                self.neg_mask_per_label_layer[layer][cur_classes[1-j]] = 1 - mask
                self.neg_mask_per_label_layer[layer][cur_classes[1-j]][:mid_size] = 1
        

    def forward(self, x, **kwargs):
        # Como condicion, las clases deben venir ordenadas de 0...n
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        x_init = x.clone().detach()
        
        goodnesses = torch.zeros(x.shape[0], self.cur_nclasses, len(self.layers), requires_grad=True).to(x.device)
        
        for l in range(self.cur_nclasses):
            x_init_c = x_init.clone().detach()
            labels = torch.full((x.shape[0],), l, dtype=torch.long).to(x.device)
            
            input_vec = self.overlay_function(x_init_c, labels)
            
            x_base = input_vec.clone().detach()
            
            for i, layer in enumerate(self.layers):
                
                if self.residual_connections and i > 0:
                    input_vec = torch.cat([input_vec.clone().detach(), x_base.clone().detach()], 1)
                else:
                    input_vec = input_vec.clone().detach()
                
                if self.masking and self.training:
                    input_vec = layer(input_vec, self.get_mask(i))
                else:
                    if self.masking:
                        input_vec = layer(input_vec, self.get_mask(i, l))
                    else:
                        input_vec = layer(input_vec)
                    
                
                goodnesses[:, l, i] = self.goodness(input_vec)
        
        # MULTIPLY on the last dimension to get the average goodness       
        return goodnesses.prod(2)

class FF_Network_MultiTask(MultiTaskModule):

    def __init__(
        self,
        residual_connections = False,
        normalize_activity = False,
        classes_on_zero = True
    ):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        
        self.task_classes = {}
        self.total_classes = []
        
        self.max_class_label = max(self.max_class_label, 0)
        
        self.residual_connections = residual_connections
        self.normalize_activity = normalize_activity
        
        self.classes_on_zero = classes_on_zero
    
    def add_layer(self, layer : nn.Module):
        self.layers.append(layer)
        
    def goodness(self, x):
        mid = x.shape[1] // 2
                
        goodness =  (x[:, :mid].pow(2).sum(1) + 1e-9)/(x.pow(2).sum(1) + 2e-9)
             
        # Done to avoid divisions by zero or problems in loss
        goodness = torch.clamp(goodness, 0.0001, 0.9999)
        
        return goodness
    
    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        
        curr_classes = experience.classes_in_this_experience
        task_labels = experience.task_labels
        
        if isinstance(task_labels, ConstantSequence):
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)
            
            self.task_classes[tid] = curr_classes
            self.total_classes += curr_classes

    def forward_single_task(self, x, task_label):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
            
        cur_classes = self.task_classes[str(task_label)]
        
        x_init = x.clone().detach()
        
        goodnesses = torch.zeros(x.shape[0], len(cur_classes), len(self.layers), requires_grad=True).to(x.device)
        
        for l in range(len(cur_classes)):
            
            x_init_c = x_init.clone().detach()
            
            if self.classes_on_zero:
                labels = torch.full((x.shape[0],), task_label*2 + cur_classes[l], dtype=torch.long).to(x.device)
            else:
                labels = torch.full((x.shape[0],), cur_classes[l], dtype=torch.long).to(x.device)
            
            input_vec = self.overlay_function(x_init_c, labels)
            
            x_base = input_vec.clone().detach()
            
            for i, layer in enumerate(self.layers):
                
                if self.residual_connections and i > 0:
                    input_vec = torch.cat([input_vec.clone().detach(), x_base.clone().detach()], 1)
                else:
                    input_vec = input_vec.clone().detach()
                    
                input_vec = layer(input_vec)
                
                goodnesses[:, l, i] = self.goodness(input_vec)
        
        # MULTIPLY on the last dimension to get the average goodness       
        return goodnesses.prod(2)