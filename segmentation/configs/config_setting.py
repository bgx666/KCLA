from torchvision import transforms
from utils import *

from datetime import datetime

class setting_config:
    """
    the config of training setting.
    """
    def __init__(self):
        self.network = 'UNet_kcla'

        self.datasets = 'isic18' 
        #self.datasets = 'isic17' 
        #self.datasets = 'kvasir'
        
        self.model_config = {
            'num_classes': 1, 
            'input_channels': 3, 
            'c_list': [8,16,24,32,48,64], 
            'bridge': True,
            'gt_ds': True,
        }
        self.criterion = GT_BceDiceLoss(wb=1, wd=1)

        self.pretrained_path = './pre_trained/'
        self.test_output_root = './test_results/'
        self.resume_dir = ''
        self.testing = True
        
        self.num_classes = 1
        self.input_channels = 3
        self.distributed = False
        self.local_rank = -1
        self.num_workers = 8
        self.seed = 42
        self.world_size = None
        self.rank = None
        self.amp = False
        self.gpu_id = '0'
        self.batch_size = 8
        self.epochs = 300

        self.work_dir = 'results/' + self.network + '_' + self.datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
        print(self.work_dir+'-------------------------------------------')

        self.print_interval = 50
        self.val_interval = 50
        self.save_interval = 30
        self.threshold = 0.5

        self.opt = 'AdamW'
        assert self.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
        self.sch = 'CosineAnnealingLR'
        assert self.sch in ['CosineAnnealingLR','ExponentialLR' ,'CosineAnnealingWarmRestarts','WP_CosineLR','WP_MultiStepLR', 'MultiStepLR', 'ReduceLROnPlateau', 'StepLR'], 'Unsupported scheduler!'
    
        self.init_config()

    def init_config(self):
        if self.datasets == 'isic18':
            self.input_size_h = 256
            self.input_size_w = 256
            self.data_path = './data/isic2018/'
        elif self.datasets == 'isic17':
            self.input_size_h = 256
            self.input_size_w = 256
            self.data_path = './data/isic2017/'
        elif self.datasets == 'kvasir':
            self.input_size_h = 256
            self.input_size_w = 256
            self.data_path = './data/Kvasir-SEG-split/'
        else:
            raise Exception('datasets in not right!')

        self.train_transformer = transforms.Compose([
            myNormalize(self.datasets, train=True),
            myToTensor(),
            myRandomHorizontalFlip(p=0.5),
            myRandomVerticalFlip(p=0.5),
            myRandomRotation(p=0.5, degree=[0, 360]), 
            myResize(self.input_size_h, self.input_size_w),
        ])
        self.test_transformer = transforms.Compose([
            myNormalize(self.datasets, train=False),
            myToTensor(),
            myResize(self.input_size_h, self.input_size_w),
        ])
    

        if self.opt == 'Adadelta':
            self.lr = 0.01
            self.rho = 0.9
            self.eps = 1e-6
            self.weight_decay = 0.05
        elif self.opt == 'Adagrad':
            self.lr = 0.01
            self.lr_decay = 0
            self.eps = 1e-10
            self.weight_decay = 0.05
        elif self.opt == 'Adam':
            self.lr = 0.001
            self.betas = (0.9, 0.999)
            self.eps = 1e-8
            self.weight_decay = 0.0001
            self.amsgrad = False
        elif self.opt == 'AdamW':
            self.lr = 0.001
            self.betas = (0.9, 0.999)
            self.eps = 1e-8
            self.weight_decay = 1e-2
            self.amsgrad = False
        elif self.opt == 'Adamax':
            self.lr = 2e-3
            self.betas = (0.9, 0.999)
            self.eps = 1e-8
            self.weight_decay = 0
        elif self.opt == 'ASGD':
            self.lr = 0.01
            self.lambd = 1e-4
            self.alpha = 0.75
            self.t0 = 1e6
            self.weight_decay = 0
        elif self.opt == 'RMSprop':
            self.lr = 1e-2
            self.momentum = 0
            self.alpha = 0.99
            self.eps = 1e-8
            self.centered = False
            self.weight_decay = 0
        elif self.opt == 'Rprop':
            self.lr = 1e-2
            self.etas = (0.5, 1.2)
            self.step_sizes = (1e-6, 50)
        elif self.opt == 'SGD':
            self.lr = 0.01
            self.momentum = 0.9
            self.weight_decay = 0.05
            self.dampening = 0
            self.nesterov = False
        
        
        if self.sch == 'StepLR':
            self.step_size = self.epochs // 5
            self.gamma = 0.5
            self.last_epoch = -1
        elif self.sch == 'MultiStepLR':
            self.milestones = [60, 120, 150]
            self.gamma = 0.1
            self.last_epoch = -1
        elif self.sch == 'ExponentialLR':
            self.gamma = 0.99
            self.last_epoch = -1
        elif self.sch == 'CosineAnnealingLR':
            self.T_max = 50
            self.eta_min = 0.00001
            self.last_epoch = -1
        elif self.sch == 'ReduceLROnPlateau':
            self.mode = 'min'
            self.factor = 0.1
            self.patience = 10
            self.threshold = 0.0001
            self.threshold_mode = 'rel'
            self.cooldown = 0
            self.min_lr = 0
            self.eps = 1e-08
        elif self.sch == 'CosineAnnealingWarmRestarts':
            self.T_0 = 50
            self.T_mult = 2
            self.eta_min = 1e-6
            self.last_epoch = -1
        elif self.sch == 'WP_MultiStepLR':
            self.warm_up_epochs = 10
            self.gamma = 0.1
            self.milestones = [125, 225]
        elif self.sch == 'WP_CosineLR':
            self.warm_up_epochs = 20
