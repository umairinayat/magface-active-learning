#!/usr/bin/env python
import sys
import os
sys.path.append("MagFace_repo")
sys.path.append("MagFace_repo/inference")

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
import mxnet as mx
from torchvision import transforms

from MagFace_repo.inference.network_inf import builder_inf


class Args:
    def __init__(self):
        self.arch = 'iresnet100'
        self.embedding_size = 512
        self.resume = 'magface_epoch_00025.pth'
        self.cpu_mode = False
        self.dist = 1


def load_bin(path, image_size=(112, 112)):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='latin1')
    
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print(f'Loading bin: {idx}/{len(issame_list) * 2}')
    
    return data_list, issame_list


def extract_embeddings(model, data_list, device, batch_size=128):
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for data in data_list:
            embeddings = []
            for i in tqdm(range(0, data.size(0), batch_size), desc="Extracting embeddings"):
                batch = data[i:i + batch_size].to(device)
                batch = (batch - 127.5) / 128.0
                feat = model(batch)
                feat = F.normalize(feat, p=2, dim=1)
                embeddings.append(feat.cpu().numpy())
            embeddings = np.concatenate(embeddings, axis=0)
            embeddings_list.append(embeddings)
    
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = embeddings / 2.0
    embeddings = F.normalize(torch.from_numpy(embeddings), p=2, dim=1).numpy()
    
    return embeddings


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def evaluate(embeddings, actual_issame, nrof_folds=10):
    from sklearn.model_selection import KFold
    
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def main():
    parser = argparse.ArgumentParser(description='MagFace Inference and Evaluation')
    parser.add_argument('--checkpoint', type=str, default='magface_epoch_00025.pth',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--data_root', type=str, default='faces_webface_112x112',
                        help='Root directory containing evaluation benchmarks')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args_main = parser.parse_args()
    
    if args_main.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args_main.device = 'cpu'
    
    device = torch.device(args_main.device)
    
    print("="*70)
    print("Loading pretrained MagFace model...")
    print("="*70)
    
    args = Args()
    args.resume = args_main.checkpoint
    args.cpu_mode = (args_main.device == 'cpu')
    
    model = builder_inf(args)
    model = model.to(device)
    if args_main.device == 'cuda':
        model = torch.nn.DataParallel(model)
    model.eval()
    
    print("Model loaded successfully!")
    
    benchmarks = ['lfw', 'agedb_30', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw']
    results = {}
    
    for benchmark in benchmarks:
        bin_path = os.path.join(args_main.data_root, f'{benchmark}.bin')
        if not os.path.exists(bin_path):
            print(f"\nSkipping {benchmark}: file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating on {benchmark.upper()}")
        print(f"{'='*70}")
        
        data_list, issame_list = load_bin(bin_path, image_size=(112, 112))
        embeddings = extract_embeddings(model, data_list, device, batch_size=args_main.batch_size)
        tpr, fpr, accuracy = evaluate(embeddings, np.array(issame_list))
        
        acc_mean = np.mean(accuracy)
        acc_std = np.std(accuracy)
        
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import auc
        
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        results[benchmark] = {
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'roc_auc': roc_auc,
            'eer': eer
        }
        
        print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"EER: {eer*100:.2f}%")
    
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    for benchmark, metrics in results.items():
        print(f"\n{benchmark.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}% ± {metrics['accuracy_std']*100:.2f}%")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  EER: {metrics['eer']*100:.2f}%")


if __name__ == '__main__':
    main()
