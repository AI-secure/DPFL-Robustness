import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

def model_copy_params(model, target_state_dict):

    own_state = model.state_dict()
    for name, param in target_state_dict.items():
        if name in own_state:
            shape = param.shape
            own_state[name].copy_(param.clone())


