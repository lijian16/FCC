import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix

import matplotlib.pyplot as plt
from utils.utils import valid_logger

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sklearn.metrics as sm 

def eval_matrix(labels, preds):
    precision, recall, f_score, _ = precision_recall_fscore_support(labels, preds, average='macro')

    cp = sm.classification_report(labels, preds)

    return precision, recall, f_score, cp

def parse_args():
    parser = argparse.ArgumentParser(description="tricks evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10_im100.yaml",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        help="decide which gpus to use",
        required=True,
        default='0',
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes):

    logger, log_file = valid_logger(cfg)

    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Sigmoid() \
        if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
        torch.nn.Softmax(dim=1)

    with torch.no_grad():
        preds = []
        out_labels = None
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()

            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)

            # cal eval matrix
            if len(preds) == 0:
                preds.append(output.detach().cpu().numpy())
                out_labels = image_labels.detach().cpu().numpy()

            else:
                preds[0] = np.append(preds[0], output.detach().cpu().numpy(), axis=0)
                out_labels = np.append(out_labels, image_labels.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    #print(out_labels)
    #print(preds)

    precision, recall, f1, cp = eval_matrix(out_labels, preds)
    logger.info("precision: {:.2f}%, recall: {:.2f}%, f1-score: {:.2f}%".format(precision*100, recall*100, f1*100))

    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )

    
    logger.info("Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%\n\n\n".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )

    logger.info("class report is \n{}".format(cp))
    

    pbar.close()
    fig = fusion_matrix.plot_confusion_matrix()
    
    save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "valid", "confusion_matrix")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=1000)

    logger.info('Save confusion matrix to {}'.format(os.path.join(save_path, 'confusion_matrix.png')))
    
    #plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cuda")
    model = Network(cfg, mode="test", num_classes=num_classes)

    model_file = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, 'models', cfg.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=cfg.TEST.TAU_NORM.USE_TAU_NORM, tau=cfg.TEST.TAU_NORM.TAU)

    model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, num_classes)
