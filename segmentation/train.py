import os
import torch
from torch.utils.data import DataLoader
import timm
from glob import glob
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.UNet_kcla import UNet_kcla
from sklearn.model_selection import train_test_split

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    
    num_runs = getattr(config, 'num_runs', 5)
    
    for run in range(1, num_runs + 1):
        print(f'\n#----------Starting Run {run}/{num_runs}----------#')
        run_checkpoint_dir = os.path.join(checkpoint_dir, f'run_{run}')
        run_log_dir = os.path.join(log_dir, f'run_{run}')
        if not os.path.exists(run_checkpoint_dir):
            os.makedirs(run_checkpoint_dir)
        if not os.path.exists(run_log_dir):
            os.makedirs(run_log_dir)
            
        set_seed(config.seed)
        global logger
        logger = get_logger(f'train_run_{run}', run_log_dir)
        global writer
        writer = SummaryWriter(os.path.join(config.work_dir, f'summary_run_{run}'))
        
        train_single_run(config, run_checkpoint_dir,run)

        torch.cuda.empty_cache()

def train_single_run(config, checkpoint_dir,run):
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network =='UNet_kcla':
        model = UNet_kcla(n_channels=model_cfg['input_channels'],n_classes=model_cfg['num_classes'])
    elif config.network == 'UNet_resnet':
        model = UNet_resnet(n_channels=model_cfg['input_channels'],n_classes=model_cfg['num_classes'])
    elif config.network == 'UNet_mrla_l':
        model = UNet_mrla_l(n_channels=model_cfg['input_channels'],n_classes=model_cfg['num_classes'])
    elif config.network == 'UNet':
        model = UNet(n_channels=model_cfg['input_channels'],n_classes=model_cfg['num_classes'])
    else: raise Exception('network in not right!')
    model = model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
                run=run
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config()
    main(config)
