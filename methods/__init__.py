from .SimCLR.simclr import SimCLRModel
from .MoCo.moco import MoCoModel
from .SimSiam.simsiam import SimSiamModel


def set_model(args):
    if args.method == 'simclr':
        return SimCLRModel(args)
    elif args.method == 'moco':
        return MoCoModel(args)
    elif args.method == 'simsiam':
        return SimSiamModel(args) 

