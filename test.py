import warnings

warnings.filterwarnings('ignore')

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import mean_absolute_error
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_mae = 0
    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp)

        save_image(res, os.path.join("result", test_data[3][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_mae += mean_absolute_error(torch.mul(res, 255).flatten(), torch.mul(tar, 255).flatten()).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size
    stat_mae /= size

    print("MAE: {}, PSNR: {}, SSIM: {}, LPIPS: {}".format(stat_mae, stat_psnr, stat_ssim, stat_lpips))


if __name__ == '__main__':
    test()
