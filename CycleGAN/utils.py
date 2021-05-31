import random, torch, os, numpy as np
from torch.autograd.grad_mode import F
import torch.nn as nn
import config
import copy


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # 预训练模型 要求模型一模一样，每一层的写法和命名都要一样  本质一样都不行
    # 完全一样的模型实现，也可能因为load和当初save时 pytorch版本不同 而导致state_dict中的key不一样
    # 例如 "initial.0.weight" 与 “initial.weight” 的区别
    model.load_state_dict(checkpoint["state_dict"], strict=False)   # 改成strict=False才能编译通过
    optimizer.load_state_dict(checkpoint["optimizer"])

    # 如果我们不这样做，将还会使用old checkpoint 中的 lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
