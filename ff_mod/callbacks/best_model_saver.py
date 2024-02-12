from ff_mod.callbacks.callback import Callback

import torch
import os

class BestModelSaver(Callback):
    def __init__(self, path, network) -> None:
        super().__init__()

        self.path = path
        self.one_saved = False
        self.best_acc = 0.0
        
        self.test_acc = 0.0
        
        self.network = network
        
    def reset(self):
        super().reset()

        self.test_acc = 0.0
        
    def on_test_batch_end(self, *args, **kwargs):
        super().on_test_batch_end(*args, **kwargs)
        
        self.test_acc += kwargs["predictions"].eq(kwargs["y"]).float().sum()
        
        
    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)
        
        if self.test_acc > self.best_acc:
            self.best_acc = self.test_acc
            
            if self.one_saved:
                
                if os.path.exists(self.path + "/best_model_old"):
                    os.remove(self.path + "/best_model_old")
                
                os.rename(self.path + "/best_model", self.path + "/best_model_old")
            
            self.network.save_network(f"{self.path}/best_model")
            self.one_saved = True
            
        self.test_acc = 0.0