def _set_seed(seed=42):
    import torch
    import torch.backends.cudnn as cudnn
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
def get_config(yaml_path):
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
