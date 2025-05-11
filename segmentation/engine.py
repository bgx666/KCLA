import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import os


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []
    with tqdm(train_loader, desc=f'Epoch {epoch} Training', leave=True) as pbar:
        for iter, data in enumerate(pbar):
            step += iter
            optimizer.zero_grad()
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            gt_pre, out = model(images)
            loss = criterion(gt_pre, out, targets)

            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())

            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            writer.add_scalar('loss', loss, global_step=step)

            if iter % config.print_interval == 0:
                log_info = f'loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                pbar.set_postfix_str(log_info)
                logger.info(f'train: epoch {epoch}, iter:{iter}, {log_info}')
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        with tqdm(test_loader, desc=f'Epoch {epoch} Validation', leave=True) as pbar:
            for data in pbar:
                img, msk = data
                img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)

                loss_list.append(loss.item())
                gts.append(msk.squeeze(1).cpu().detach().numpy())
                if type(out) is tuple:
                    out = out[0]
                out = out.squeeze(1).cpu().detach().numpy()
                preds.append(out)
                pbar.set_postfix_str(f'loss: {np.mean(loss_list):.4f}')

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        tqdm.write(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        tqdm.write(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None,
                    run=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing', leave=True) as pbar:
            for i, data in enumerate(pbar):
                img, msk = data
                img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

                gt_pre, out = model(img)
                loss = criterion(gt_pre, out, msk)

                loss_list.append(loss.item())
                msk = msk.squeeze(1).cpu().detach().numpy()
                gts.append(msk)
                if type(out) is tuple:
                    out = out[0]
                out = out.squeeze(1).cpu().detach().numpy()
                preds.append(out)
                if i % config.save_interval == 0:
                    output_dir = os.path.join(config.work_dir, 'outputs', f'run_{run}' if run is not None else '')
                    os.makedirs(output_dir, exist_ok=True)
                    save_imgs(img, msk, out, i, output_dir, config.datasets, config.threshold, test_data_name=test_data_name)
                pbar.set_postfix_str(f'loss: {np.mean(loss_list):.4f}')

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            tqdm.write(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        tqdm.write(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
