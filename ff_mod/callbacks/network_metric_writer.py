from ff_mod.callbacks.callback import Callback
from sklearn.metrics import confusion_matrix

import seaborn as sns

class NetworkMetricWriter(Callback):
    def __init__(self, tensorboard = None) -> None:
        super().__init__()

        self.train_len = 0.0
        
        self.train_total_len = 0.0
                
        self.tensorboard = tensorboard
        
    def reset(self):        
        self.train_len = 0.0
        self.test_len = 0.0
    
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        self.train_len += 1
        self.tensorboard.add_scalar("loss", kwargs["loss"], self.train_total_len + self.train_len)
        
    def on_test_batch_end(self, *args, **kwargs):
        super().on_test_batch_end(*args, **kwargs)
            
    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)
        
        self.train_total_len += self.train_len
        self.train_len = 0.0
        
    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)
        