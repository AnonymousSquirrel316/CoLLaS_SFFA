from ff_mod.networks_cl.network.ff_network import FF_Network, FF_Layer, FF_Network_MultiTask, FF_Network_Incremental
from ff_mod.overlay import AppendToEndOverlay
from ff_mod.loss.loss import AvalancheBCELoss

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import avalanche.models as mods
from avalanche.models import BaseModel
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, LFLPlugin, GEMPlugin, SynapticIntelligencePlugin, MASPlugin
from avalanche.benchmarks.classic import SplitMNIST, SplitFMNIST

class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        """
        :param initial_out_features: if >0 override output size and build an
            IncrementalClassifier with `initial_out_features` units as first.
        """
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)

        
        if initial_out_features > 0:
            self.classifier = mods.IncrementalClassifier(in_features=hidden_size, initial_out_features=initial_out_features, masking=False)
        elif output_size > 0:
            self.classifier = nn.Linear(hidden_size, output_size)
            

    def forward(self, x, task_labels = None):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        
        if task_labels is not None:
            x = self.classifier(x, task_labels)
        else:
            x = self.classifier(x)
        
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        return self.features(x)
    

class MultiHeadMLP(mods.MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 drop_rate=0, relu_act=True):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = mods.MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x

def get_ff_model(type = None):
    overlay_function = AppendToEndOverlay(100, 10)

    layer_1 = FF_Layer(884, 1400, bias=True)
    layer_2 = FF_Layer(1400 + 884, 1400, bias=True)

    if type == "ci":
        model = FF_Network_Incremental(2, True, True)
    elif type == "di":
        model = FF_Network(True, True)
        model.total_classes = 2
    else:
        model = FF_Network_MultiTask(2, True, True)

    model.add_layer(layer_1)
    model.add_layer(layer_2)

    model.overlay_function = overlay_function

    model.cuda()
    model.train()

        
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = AvalancheBCELoss()
    
    return model, optimizer, criterion

def get_sd_model(type):
    if type == "ci":
        model = MLP(hidden_size=1400, hidden_layers=2, initial_out_features=2)
    elif type == "di": # Domain Incremental
        model = MLP(hidden_size=1400, hidden_layers=2, output_size=2)
    else:
        model = MultiHeadMLP(hidden_size=1400, hidden_layers=2)
    model.cuda()
        
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    return model, optimizer, criterion
    
    
def test(model, optmizer, criterion, plugin, type : str = 'ci', seed : int = 0):
    # Type can be ci, di, ti
    
    if plugin is not None:
        strategy = SupervisedTemplate(
            model, optmizer, criterion,
            train_mb_size=512, train_epochs=2, eval_mb_size=512,
            device="cuda:0", eval_every=0, plugins=[plugin])
    else:
        strategy = SupervisedTemplate(
            model, optmizer, criterion,
            train_mb_size=512, train_epochs=2, eval_mb_size=512,
            device="cuda:0", eval_every=0, plugins=[])

    if seed <= 10:
        if type == 'ci':
            # Class Incremental
            benchmark = SplitMNIST(n_experiences=5, seed=seed, class_ids_from_zero_from_first_exp=True, return_task_id=False)
        elif type == 'di':
            # Domain Incremental
            benchmark = SplitMNIST(n_experiences=5, seed=seed, class_ids_from_zero_in_each_exp=True, return_task_id=False)
        else:
            # Task Incremental
            benchmark = SplitMNIST(n_experiences=5, seed=seed, class_ids_from_zero_in_each_exp=True, return_task_id=True)
    else:
        if type == 'ci':
            # Class Incremental
            benchmark = SplitFMNIST(n_experiences=5, seed=seed, class_ids_from_zero_from_first_exp=True, return_task_id=False)
        elif type == 'di':
            # Domain Incremental
            benchmark = SplitFMNIST(n_experiences=5, seed=seed, class_ids_from_zero_in_each_exp=True, return_task_id=False)
        else:
            # Task Incremental
            benchmark = SplitFMNIST(n_experiences=5, seed=seed, class_ids_from_zero_in_each_exp=True, return_task_id=True)
            
    # TRAINING LOOP
    results = []
    for i, experience in enumerate(benchmark.train_stream):

        strategy.train(experience)

        res = strategy.eval(benchmark.test_stream[:i+1])
        
        acc = 0.0
        for j in range(i+1):
            # Top1_Acc_Exp/eval_phase/test_stream/Task00{j}/Exp00{j}
            if type == 'ti':
                acc += res[f'Top1_Acc_Exp/eval_phase/test_stream/Task00{j}/Exp00{j}']
            else:
                # Check if "FF_Model" in model class name
                acc += res[f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{j}']
            
        acc /= (i+1)
        results.append(acc)
        
    return results

# Reinitialize plugins
def get_plugin(type):
    if  type == "syn":
        return SynapticIntelligencePlugin(si_lambda=1000,eps=0.1)
    elif type == "ewc":
        return EWCPlugin(ewc_lambda=100000)
    elif type == "rep":
        return ReplayPlugin(mem_size=500)
    elif type == "gem":
        return GEMPlugin(20, 0.5)
    elif type == "mas":
        return MASPlugin()
    elif type == "lfl":
        return LFLPlugin(lambda_e = 0.0001)
    else:
        return None

#for seed in [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
for seed in [1,2,6,7,8,9,10,11,20]:
    print(f"Seed {seed}")
    final_results = {}

    for task_type in ['ci', 'di', 'ti']:
        print(f'Task type: {task_type}')
        for model_type in ['ff','sd']:
            print(f'Model type: {model_type}')
            #for plugin in ["mas", "rep", "ewc", "syn", "gem"]:
            for plugin in ["none", "mas", "rep", "ewc", "syn", "gem"]:
                print("\n\n")
                print(f'{task_type}_{model_type}_{plugin}')
                print("\n\n")
                plugin_str = plugin
                
                model, optimizer, criterion = get_ff_model(task_type) if model_type == 'ff' else get_sd_model(task_type)
                
                plugin = get_plugin(plugin)
                
                final_results[f'{task_type}_{model_type}_{plugin_str}'] = test(model, optimizer, criterion, plugin, task_type, seed)


    # Save dict to file as json

    import json

    with open(f'experiments/train/results_seed_{seed}.json', 'w') as fp:
        json.dump(final_results, fp)