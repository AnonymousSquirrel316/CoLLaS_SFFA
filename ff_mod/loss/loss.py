from typing import Any, Optional
import torch


class ProbabilityBCELoss:
    """
        BCE Loss for probability values of the goodness vector
    """
    def __init__(self, use_normalization = True):
        super().__init__()
        
        self.use_normalization = use_normalization
        
        self.writer = None
        self.writer_step = 0
        
    def __call__(self, g_pos, g_neg, **kwargs):
            
        g_pos_r = kwargs['latent_pos'].mean()
        g_neg_r = kwargs['latent_neg'].mean()
        
        loss_1 = -torch.log(torch.cat([g_pos, (1-g_neg)])).mean()
        loss_2 = (1+torch.exp(-10*(g_pos_r+g_neg_r)))
        
        if self.writer is not None:
            self.writer_step += 1
            
            self.writer.add_scalar("Loss/loss_1", loss_1, self.writer_step)
            self.writer.add_scalar("Loss/loss_2", loss_2, self.writer_step)
            self.writer.add_scalar("Latent/Mean_pos", kwargs['latent_pos'].mean(), self.writer_step)
            self.writer.add_scalar("Latent/Mean_pos_squared", kwargs['latent_pos'].pow(2).mean(), self.writer_step)
            self.writer.add_scalar("Latent/Mean_neg", kwargs['latent_neg'].mean(), self.writer_step)
            self.writer.add_scalar("Latent/Max_pos", kwargs['latent_pos'].max(), self.writer_step)
            self.writer.add_scalar("Latent/Max_neg", kwargs['latent_neg'].max(), self.writer_step)
            self.writer.add_scalar("Latent/Min_pos", kwargs['latent_pos'].min(), self.writer_step)
            self.writer.add_scalar("Latent/Min_neg", kwargs['latent_neg'].min(), self.writer_step)
            self.writer.add_scalar("Goodness/Mean_pos", g_pos.mean(), self.writer_step)
            self.writer.add_scalar("Goodness/Mean_neg", g_neg.mean(), self.writer_step)
            self.writer.add_scalar("Goodness/Max_pos", g_pos.max(), self.writer_step)
            self.writer.add_scalar("Goodness/Max_neg", g_neg.max(), self.writer_step)
            self.writer.add_scalar("Goodness/Min_pos", g_pos.min(), self.writer_step)
            self.writer.add_scalar("Goodness/Min_neg", g_neg.min(), self.writer_step)
        
        if self.use_normalization:
            return loss_1 * loss_2
        return loss_1


class AvalancheBCELoss:
    """
        BCE Loss for probability values of the goodness vector
    """
    def __init__(self) -> None:
        self.bce = torch.nn.BCELoss()
        self.batch_size = 512
        
    def __call__(self, x, y):
        # X shape (batch, labels, layers)
        # Y shape (batch, labels)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        
        total_classes = x.shape[1]
        layer_size = x.shape[2]
        
        if y.max() >= total_classes:
            raise Exception(f"Y max value {y.max()} is greater than total classes {total_classes}")
        
        # Y shape (batch, labels)
        y = torch.nn.functional.one_hot(y, num_classes=total_classes).float()
        
        # Y shape (batch, labels, layer)
        y = y.unsqueeze(2).repeat_interleave(layer_size, dim=2)
        #print(f" Min {x.min()} Max {x.max()} Mean {x.mean()} Median {x.median()}")
        return self.bce(x.flatten(), y.flatten())

class VectorBCELoss:
    def __init__(
            self,
            threshold = 0.5,
            alpha = 1.0,
            beta: Optional[float] = None,
            negative_threshold : Optional[float] = None
        ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        
        if beta is None:
            self.beta = alpha
        if negative_threshold is None:
            self.negative_threshold = threshold
    
    def __call__(self, g_pos, g_neg, **kwargs):
        
        loss = torch.nn.functional.softplus(torch.cat([
                    self.alpha*(-g_pos + self.threshold),
                    self.beta*(g_neg - self.negative_threshold)])).mean()
        return loss

class MeanBCELoss:
    def __init__(self, threshold: float = 0.5, alpha = 1.0, beta: Optional[float] = None):
        super().__init__()
        
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        
        if beta is None:
            self.beta = alpha
        
    def __call__(self, g_pos, g_neg, **kwargs):
        loss_pos = torch.log(1 + torch.exp(self.alpha * (-g_pos.mean() + self.threshold)))
        loss_neg = torch.log(1 + torch.exp(self.beta * (g_neg.mean() - self.threshold)))
        return loss_pos+loss_neg

class SymbaLoss:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, g_pos, g_neg, **kwargs):
        return torch.nn.functional.softplus(self.alpha * (g_neg.mean() - g_pos.mean()))