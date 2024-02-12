from ff_mod.dataloader.KMNIST import KMNIST_Dataloader
from ff_mod.dataloader.EMNIST import EMNIST_Dataloader
from ff_mod.dataloader.FashionMNIST import FashionMNIST_Dataloader
from ff_mod.dataloader.MNIST import MNIST_Dataloader

from ff_mod.dataloader.CIFAR import CIFAR10_Dataloader


from ff_mod.dataloader.base import DataLoaderExtractor

class DataloaderFactory:
    # TODO Refactor this name
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, dataset : str = None, batch_size : int = 512, **kwargs) -> DataLoaderExtractor:
        if dataset is None:
            raise ValueError("dataset is None")
        

        if dataset == "kmnist":
            return KMNIST_Dataloader(batch_size=batch_size, **kwargs)
        elif dataset == "fashion_mnist":
            return FashionMNIST_Dataloader(batch_size=batch_size, **kwargs)
        elif dataset == "emnist":
            return EMNIST_Dataloader(batch_size=batch_size, **kwargs)
        elif dataset == "mnist":
            return MNIST_Dataloader(batch_size=batch_size, **kwargs)
        elif dataset == "cifar10":
            return CIFAR10_Dataloader(batch_size=batch_size, **kwargs)
        
        raise ValueError("The dataloader is not valid.")
    
    def get_num_classes(self, dataset) -> int:
        
        if dataset == "kmnist":
            return 10
        elif dataset == "fashion_mnist":
            return 10
        elif dataset == "emnist":
            return 26
        elif dataset == "mnist":
            return 10
        elif dataset == "cifar10":
            return 10
    
    def get_valid_dataloaders(self):
        return ["kmnist", "fashion_mnist", "emnist", "mnist"]
    
    def get_mix_dataloader(self, datasets : list[str], batch_size : int = 512):
        raise NotImplementedError("Mix dataloader is not implemented yet.")