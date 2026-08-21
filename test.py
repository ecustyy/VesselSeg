import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img,group_images,concat_result
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from lib.postprocess import connect_vessel_by_direction

setpu_seed(2021)

class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)

    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        for i in range(self.pred_imgs.shape[0]):
            pretep = connect_vessel_by_direction(
                self.pred_imgs[i][0],
                distance_threshold=8,
                angle_threshold=5
            )
            self.pred_imgs[i][0] = pretep   #*self.pred_imgs[i][0]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            # Save composited result
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_masks[i])
            save_img(total_img, join(self.save_img_path, "Result_" + img_name_list[i] + '.png'))

            # Save grayscale prediction separately
            gray_path = join(self.save_img_path, "Gray_" + img_name_list[i] + '.png')
            save_img(self.pred_imgs[i][0] * 255, gray_path)

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)

if __name__ == '__main__':
    args = parse_args()
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net = models.UNetFamily.U_Net(3,2).to(device)    #初始化模型
    # net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    cudnn.benchmark = True

    # Load checkpoint   #加载模型
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'top1_model.pth'))
    net.load_state_dict(checkpoint['net'], strict=False)

    eval = Test(args)   #初始化的时候已经根据arg设置了数据加载器
    eval.inference(net) #推理出每个patch的结果
    print(eval.evaluate())  #恢复成图片并评估指标
    eval.save_segmentation_result() #保存结果图片
