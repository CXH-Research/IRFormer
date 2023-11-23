import warnings

import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *

warnings.filterwarnings('ignore')


def train():
    # Accelerate
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
    device = accelerator.device

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("ir", config=config)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                             drop_last=False, pin_memory=True)
    val_dataset = get_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Loss
    model = Model()
    criterion_psnr = torch.nn.SmoothL1Loss()

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    start_epoch = 1
    best_epoch = 1
    best_psnr = 0

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            # get the inputs; data is a list of [target, input, filename]
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)

            train_loss = loss_psnr + 0.2 * loss_ssim

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            mae = 0
            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = data[0].contiguous()
                tar = data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255)).item()

            psnr /= size
            ssim /= size
            mae /= size

            if psnr > best_psnr:
                # save model
                best_epoch = epoch
                best_psnr = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "MAE": mae,
            }, step=epoch)

            if accelerator.is_local_main_process:
                print(
                    "epoch: {}, MAE:{}, PSNR: {}, SSIM: {}, best PSNR: {}, best epoch: {}"
                    .format(epoch, mae, psnr, ssim, best_psnr, best_epoch))

    accelerator.end_training()


if __name__ == '__main__':
    train()
