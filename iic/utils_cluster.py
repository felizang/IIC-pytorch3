from __future__ import print_function

import os
import sys
from datetime import datetime
from colorsys import hsv_to_rgb

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision
import torchvision.transforms.functional as tf
# import itertools
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, ConcatDataset

from sklearn import metrics
from sklearn.cluster import KMeans
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

from PIL import Image
# from torch.autograd import Variable

'''
from code.utils.cluster.data import _cifar100_to_cifar20, \
  _create_dataloaders, _create_mapping_loader
from code.utils.cluster.eval_metrics import _hungarian_match, _acc
from code.utils.cluster.transforms import sobel_make_transforms, \
  greyscale_make_transforms
from code.utils.cluster.transforms import sobel_process

from code.datasets.clustering.truncated_dataset import TruncatedDataset
from code.utils.cluster.transforms import sobel_make_transforms, \
  greyscale_make_transforms
from code.utils.semisup.dataset import TenCropAndFinish
from general import reorder_train_deterministic

from .IID_losses import IID_loss
from .eval_metrics import _hungarian_match, _original_match, _acc
from .transforms import sobel_process

from .cluster_eval import _acc, _original_match, _hungarian_match

from code.utils.cluster.cluster_eval import _get_assignment_data_matches, \
  _clustering_get_data
'''

from torch.optim import Adam

_opt_dict = {"Adam": Adam}


class TruncatedDataset(Dataset):
    def __init__(self, base_dataset, pc):
        self.base_dataset = base_dataset
        self.len = int(len(self.base_dataset) * pc)
        # also shuffles. Ok because not using for train
        self.random_order = np.random.choice(len(self.base_dataset), size=self.len,
                                             replace=False)

    def __getitem__(self, item):
        assert (item < self.len)

        return self.base_dataset.__getitem__(self.random_order[item])
        # return self.base_dataset.__getitem__(item)

    def __len__(self):
        return self.len


def iid_loss(x_out, x_tf_out, lamb=1.0, eps=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < eps).data] = eps
    p_j[(p_j < eps).data] = eps
    p_i[(p_i < eps).data] = eps

    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = loss.sum()

    loss_trace = 1 - torch.trace(p_i_j).item()
    loss_diff = torch.abs(x_out - x_tf_out)
    loss_diff = loss_diff.sum() / x_out.size(0)
    loss_diff = loss_diff / x_out.size(1)

    loss += 0.1*loss_diff

    # loss_no_lamb = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))

    # loss_no_lamb = loss_no_lamb.sum()
    # loss_no_lamb += 0.5*(1 - torch.trace(p_i_j).item())

    return loss, loss_diff, p_i_j  # loss_no_lamb


def iic_loss(x_out, x_tf_out, lamb=1.0, eps=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1)
    p_j = p_i_j.sum(dim=0)

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < eps).data] = eps
    p_j[(p_j < eps).data] = eps
    p_i[(p_i < eps).data] = eps

    hz = - p_i * torch.log(p_i)
    hzs = hz.sum()

    conp = p_i_j / p_j
    hzz = - conp * torch.log(conp)
    hzzs = hzz.sum()

    loss = hzs - hzzs

    loss_trace = 1 - torch.trace(p_i_j).item()
    loss += loss_trace

    # loss_no_lamb = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))

    # loss_no_lamb = loss_no_lamb.sum()
    # loss_no_lamb += 0.5*(1 - torch.trace(p_i_j).item())

    return loss, torch.as_tensor(loss_trace)  # loss_no_lamb


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def make_triplets_data(config):
    target_transform = None

    if "CIFAR" in config.dataset:
        config.train_partitions_head_A = [True, False]
        config.train_partitions_head_B = config.train_partitions_head_A

        config.mapping_assignment_partitions = [True, False]
        config.mapping_test_partitions = [True, False]

        if config.dataset == "CIFAR10":
            dataset_class = torchvision.datasets.CIFAR10
        elif config.dataset == "CIFAR100":
            dataset_class = torchvision.datasets.CIFAR100
        elif config.dataset == "CIFAR20":
            dataset_class = torchvision.datasets.CIFAR100
            target_transform = _cifar100_to_cifar20
        else:
            assert False

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "STL10":
        assert config.mix_train
        if not config.stl_leave_out_unlabelled:
            print("adding unlabelled data for STL10")
            config.train_partitions_head_A = ["train+unlabeled", "test"]
        else:
            print("not using unlabelled data for STL10")
            config.train_partitions_head_A = ["train", "test"]

        config.train_partitions_head_B = ["train", "test"]

        config.mapping_assignment_partitions = ["train", "test"]
        config.mapping_test_partitions = ["train", "test"]

        dataset_class = torchvision.datasets.STL10

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "MNIST":
        config.train_partitions_head_A = [True, False]
        config.train_partitions_head_B = config.train_partitions_head_A

        config.mapping_assignment_partitions = [True, False]
        config.mapping_test_partitions = [True, False]

        dataset_class = torchvision.datasets.MNIST

        tf1, tf2, tf3 = greyscale_make_transforms(config)

    else:
        assert False

    dataloaders = \
        _create_dataloaders(config, dataset_class, tf1, tf2,
                            partitions=config.train_partitions_head_A,
                            target_transform=target_transform)

    dataloader_original = dataloaders[0]
    dataloader_positive = dataloaders[1]

    shuffled_dataloaders = \
        _create_dataloaders(config, dataset_class, tf1, tf2,
                            partitions=config.train_partitions_head_A,
                            target_transform=target_transform,
                            shuffle=True)

    dataloader_negative = shuffled_dataloaders[0]

    # since this is fully unsupervised, assign dataloader = test dataloader
    dataloader_test = \
        _create_mapping_loader(config, dataset_class, tf3,
                               partitions=config.mapping_test_partitions,
                               target_transform=target_transform)

    return dataloader_original, dataloader_positive, dataloader_negative, dataloader_test


def triplets_get_data(config, net, dataloader, sobel):
    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda()
    flat_preds_all = torch.zeros((num_batches * config.batch_sz),
                                 dtype=torch.int32).cuda()

    num_test = 0
    for b_i, batch in enumerate(dataloader):
        imgs = batch[0].cuda()

        if sobel:
            imgs = sobel_process(imgs, config.include_rgb)

        flat_targets = batch[1]

        with torch.no_grad():
            x_outs = net(imgs)

        assert (x_outs.shape[1] == config.output_k)
        assert (len(x_outs.shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        flat_preds_curr = torch.argmax(x_outs, dim=1)  # along output_k
        flat_preds_all[start_i:(start_i + num_test_curr)] = flat_preds_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_preds_all = flat_preds_all[:num_test]
    flat_targets_all = flat_targets_all[:num_test]

    return flat_preds_all, flat_targets_all


def triplets_get_data_kmeans_on_features(config, net, dataloader, sobel):
    # ouput of network is features (not softmaxed)
    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda()
    features_all = np.zeros((num_batches * config.batch_sz, config.output_k),
                            dtype=np.float32)

    num_test = 0
    for b_i, batch in enumerate(dataloader):
        imgs = batch[0].cuda()

        if sobel:
            imgs = sobel_process(imgs, config.include_rgb)

        flat_targets = batch[1]

        with torch.no_grad():
            x_outs = net(imgs)

        assert (x_outs.shape[1] == config.output_k)
        assert (len(x_outs.shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        features_all[start_i:(start_i + num_test_curr), :] = x_outs.cpu().numpy()
        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    features_all = features_all[:num_test, :]
    flat_targets_all = flat_targets_all[:num_test]

    kmeans = KMeans(n_clusters=config.gt_k).fit(features_all)
    flat_preds_all = torch.from_numpy(kmeans.labels_).cuda()

    assert (flat_targets_all.shape == flat_preds_all.shape)
    assert (max(flat_preds_all) < config.gt_k)

    return flat_preds_all, flat_targets_all


def triplets_eval(config, net, dataloader_test, sobel):
    net.eval()

    if not config.kmeans_on_features:
        flat_preds_all, flat_targets_all = triplets_get_data(config, net,
                                                             dataloader_test, sobel)
        assert (config.output_k == config.gt_k)
    else:
        flat_preds_all, flat_targets_all = triplets_get_data_kmeans_on_features(
            config, net, dataloader_test, sobel)

    num_samples = flat_preds_all.shape[0]
    assert (num_samples == flat_targets_all.shape[0])

    net.train()

    match = _hungarian_match(flat_preds_all, flat_targets_all,
                             preds_k=config.gt_k,
                             targets_k=config.gt_k)

    found = torch.zeros(config.gt_k)  # sanity
    reordered_preds = torch.zeros(num_samples,
                                  dtype=flat_preds_all.dtype).cuda()

    for pred_i, target_i in match:
        reordered_preds[flat_preds_all == pred_i] = target_i
        found[pred_i] = 1

    assert (found.sum() == config.gt_k)  # each class must get mapped

    mass = np.zeros((1, config.gt_k))
    per_class_acc = np.zeros((1, config.gt_k))
    for c in range(config.gt_k):
        flags = (reordered_preds == c)
        actual = (flat_targets_all == c)
        mass[0, c] = flags.sum().item()
        per_class_acc[0, c] = (flags * actual).sum().item()

    acc = _acc(reordered_preds, flat_targets_all, config.gt_k)

    is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))
    config.epoch_acc.append(acc)

    if config.masses is None:
        assert (config.per_class_acc is None)
        config.masses = mass
        config.per_class_acc = per_class_acc
    else:
        config.masses = np.concatenate((config.masses, mass), axis=0)
        config.per_class_acc = np.concatenate(
          (config.per_class_acc, per_class_acc), axis=0)

    return is_best


def triplets_loss(outs_orig, outs_pos, outs_neg):
    orig = nnf.log_softmax(outs_orig, dim=1)
    pos = nnf.softmax(outs_pos, dim=1)
    neg = nnf.softmax(outs_neg, dim=1)

    # loss is minimised
    return nnf.kl_div(orig, pos, reduction="elementwise_mean") \
        - nnf.kl_div(orig, neg, reduction="elementwise_mean")


def _original_match(flat_preds, flat_targets, preds_k, targets_k):
    # map each output channel to the best matching ground truth (many to one)

    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor) and
            flat_preds.is_cuda and flat_targets.is_cuda)

    out_to_gts = {}
    out_to_gts_scores = {}
    for out_c in range(preds_k):
        for gt_c in range(targets_k):
            # the amount of out_c at all the gt_c samples
            tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score

    return list(out_to_gts.items())


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor) and
            flat_preds.is_cuda and flat_targets.is_cuda)

    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match_row, match_col = linear_sum_assignment(num_samples - num_correct)
    # print(f'match size {match_row.shape}')

    # return as list of tuples, out_c to gt_c
    res = []
    # for out_c, gt_c in match:
    #    res.append((out_c, gt_c))
    for i in range(match_row.shape[0]):
        res.append((match_row[i], match_col[i]))

    return res


def _acc(preds, targets, num_k, verbose=0):
    assert (isinstance(preds, torch.Tensor) and
            isinstance(targets, torch.Tensor) and
            preds.is_cuda and targets.is_cuda)

    if verbose >= 2:
        print("calling acc...")

    assert (preds.shape == targets.shape)
    assert (preds.max() < num_k and targets.max() < num_k)

    acc = int((preds.to(torch.int) == targets.to(torch.int)).sum()) / float(preds.shape[0])

    return acc


def _nmi(preds, targets):
    return metrics.normalized_mutual_info_score(targets, preds)


def _ari(preds, targets):
    return metrics.adjusted_rand_score(targets, preds)


def get_opt(name):
    return _opt_dict[name]


def config_to_str(config):
    attrs = vars(config)
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val


def update_lr(optimiser, lr_mult=0.1):
    for param_group in optimiser.param_groups:
        param_group['lr'] *= lr_mult
    return optimiser


def reorder_train_deterministic(dataset):
    assert (isinstance(dataset, torchvision.datasets.STL10))
    assert (dataset.split == "train+unlabeled")

    # move first 5k into rest of 100k
    # one every 20 images
    assert (dataset.data.shape == (105000, 3, 96, 96))

    # 0, 5000...5019, 1, 5020...5039, 2, ... 4999, 104980 ... 104999
    ids = []
    for i in range(5000):
        ids.append(i)
        ids += range(5000 + i * 20, 5000 + (i + 1) * 20)

    dataset.data = dataset.data[ids]
    assert (dataset.data.shape == (105000, 3, 96, 96))
    dataset.labels = dataset.labels[ids]
    assert (dataset.labels.shape == (105000,))

    return dataset


def print_weights_and_grad(net):
    print("---------------")
    for n, p in net.named_parameters():
        print("%s abs: min %f max %f max grad %f" %
              (n, torch.abs(p.data).min().item(), torch.abs(p.data).max().item(),
               torch.abs(p.grad).max().item()))
    print("---------------")


def nice(dict_):
    res = ""
    for k, v in dict_.items():
        res += ("\t%s: %s\n" % (k, v))
    return res


def custom_greyscale_to_tensor(include_rgb):
    def _inner(img):
        grey_img_tensor = tf.to_tensor(tf.to_grayscale(img, num_output_channels=1))
        result = grey_img_tensor  # 1, 96, 96 in [0, 1]
        assert (result.size(0) == 1)

        if include_rgb:  # greyscale last
            img_tensor = tf.to_tensor(img)
            result = torch.cat([img_tensor, grey_img_tensor], dim=0)
            assert (result.size(0) == 4)

        return result

    return _inner


def custom_cutout(min_box=None, max_box=None):
    def _inner(img):
        w, h = img.size

        # find left, upper, right, lower
        box_sz = np.random.randint(min_box, max_box + 1)
        half_box_sz = int(np.floor(box_sz / 2.))
        x_c = np.random.randint(half_box_sz, w - half_box_sz)
        y_c = np.random.randint(half_box_sz, h - half_box_sz)
        box = (
            x_c - half_box_sz, y_c - half_box_sz, x_c + half_box_sz,
            y_c + half_box_sz)

        img.paste(0, box=box)
        return img

    return _inner


def sobel_process(imgs, include_rgb, using_ir=False):
    bn, c, h, w = imgs.size()

    if not using_ir:
        if not include_rgb:
            assert (c == 1)
            grey_imgs = imgs
        else:
            assert (c == 4)
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            rgb_imgs = imgs[:, :3, :, :]
    else:
        if not include_rgb:
            assert (c == 2)
            grey_imgs = imgs[:, 0, :, :].unsqueeze(1)  # underneath IR
            ir_imgs = imgs[:, 1, :, :].unsqueeze(1)
        else:
            assert (c == 5)
            rgb_imgs = imgs[:, :3, :, :]
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            ir_imgs = imgs[:, 4, :, :].unsqueeze(1)

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    # conv1.weight = nn.Parameter(
    #     torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
    conv1.weight = nn.Parameter(
        torch.from_numpy(sobel1).cuda().float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    # dx = conv1(Variable(grey_imgs)).data
    dx = conv1(grey_imgs).data

    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    # conv2.weight = nn.Parameter(
    #     torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
    conv2.weight = nn.Parameter(
        torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    # dy = conv2(Variable(grey_imgs)).data
    dy = conv2(grey_imgs).data

    sobel_imgs = torch.cat([dx, dy], dim=1)
    assert (sobel_imgs.shape == (bn, 2, h, w))

    if not using_ir:
        if include_rgb:
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)
            assert (sobel_imgs.shape == (bn, 5, h, w))
    else:
        if include_rgb:
            # stick both rgb and ir back on in right order (sobel sandwiched inside)
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs, ir_imgs], dim=1)
        else:
            # stick ir back on in right order (on top of sobel)
            sobel_imgs = torch.cat([sobel_imgs, ir_imgs], dim=1)

    return sobel_imgs


def per_img_demean(img):
    assert (len(img.size()) == 3 and img.size(0) == 3)  # 1 RGB image, tensor
    mean = img.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / (img.size(1) * img.size(2))

    return img - mean  # expands


def sobel_make_transforms(config, random_affine=False,
                          cutout=False,
                          cutout_p=None,
                          cutout_max_box=None,
                          affine_p=None):
    tf1_list = []
    tf2_list = []
    tf3_list = []
    if config.crop_orig:
        tf1_list += [
            torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                              config.rand_crop_sz]))),
            torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                          config.input_sz]))),
        ]
        tf3_list += [
            torchvision.transforms.CenterCrop(tuple(np.array([config.rand_crop_sz,
                                                              config.rand_crop_sz]))),
            torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                          config.input_sz]))),
        ]

    print(
        "(_sobel_multioutput_make_transforms) config.include_rgb: %s" %
        config.include_rgb)
    tf1_list.append(custom_greyscale_to_tensor(config.include_rgb))
    tf3_list.append(custom_greyscale_to_tensor(config.include_rgb))

    if config.fluid_warp:
        # 50-50 do rotation or not
        print("adding rotation option for imgs_tf: %d" % config.rot_val)
        tf2_list += [torchvision.transforms.RandomApply(
            [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

        imgs_tf_crops = []
        for crop_sz in config.rand_crop_szs_tf:
            print("adding crop size option for imgs_tf: %d" % crop_sz)
            imgs_tf_crops.append(torchvision.transforms.RandomCrop(crop_sz))
        tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]
    else:
        # default
        tf2_list += [
            torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                              config.rand_crop_sz])))]

    if random_affine:
        print("adding affine with p %f" % affine_p)
        tf2_list.append(torchvision.transforms.RandomApply(
            [torchvision.transforms.RandomAffine(18,
                                                 scale=(0.9, 1.1),
                                                 translate=(0.1, 0.1),
                                                 shear=10,
                                                 resample=Image.BILINEAR,
                                                 fillcolor=0)], p=affine_p)
        )

    assert (not (cutout and config.cutout))
    if cutout or config.cutout:
        assert (not config.fluid_warp)
        if config.cutout:
            cutout_p = config.cutout_p
            cutout_max_box = config.cutout_max_box

        print("adding cutout with p %f max box %f" % (cutout_p,
                                                      cutout_max_box))
        # https://github.com/uoguelph-mlrg/Cutout/blob/master/images
        # /cutout_on_cifar10.jpg
        tf2_list.append(
            torchvision.transforms.RandomApply(
                [custom_cutout(min_box=int(config.rand_crop_sz * 0.2),
                               max_box=int(config.rand_crop_sz *
                                           cutout_max_box))],
                p=cutout_p)
        )
    else:
        print("not using cutout")

    tf2_list += [
        torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                      config.input_sz]))),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                           saturation=0.4, hue=0.125)
    ]

    tf2_list.append(custom_greyscale_to_tensor(config.include_rgb))

    if config.demean:
        print("demeaning data")
        tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
    else:
        print("not demeaning data")

    if config.per_img_demean:
        print("per image demeaning data")
        tf1_list.append(per_img_demean)
        tf2_list.append(per_img_demean)
        tf3_list.append(per_img_demean)
    else:
        print("not per image demeaning data")

    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)

    return tf1, tf2, tf3


def greyscale_make_transforms(config):
    tf1_list = []
    tf3_list = []
    tf2_list = []

    # tf1 and 3 transforms
    if config.crop_orig:
        # tf1 crop
        if config.tf1_crop == "random":
            print("selected random crop for tf1")
            tf1_crop_fn = torchvision.transforms.RandomCrop(config.tf1_crop_sz)
        elif config.tf1_crop == "centre_half":
            print("selected centre_half crop for tf1")
            tf1_crop_fn = torchvision.transforms.RandomChoice([
              torchvision.transforms.RandomCrop(config.tf1_crop_sz),
              torchvision.transforms.CenterCrop(config.tf1_crop_sz)
            ])
        elif config.tf1_crop == "centre":
            print("selected centre crop for tf1")
            tf1_crop_fn = torchvision.transforms.CenterCrop(config.tf1_crop_sz)
        else:
            assert False
        tf1_list += [tf1_crop_fn]

        if config.tf3_crop_diff:
            print("tf3 crop size is different to tf1")
            tf3_list += [torchvision.transforms.CenterCrop(config.tf3_crop_sz)]
        else:
            print("tf3 crop size is same as tf1")
            tf3_list += [torchvision.transforms.CenterCrop(config.tf1_crop_sz)]

    tf1_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]
    tf3_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]

    # tf2 transforms
    if config.rot_val > 0:
        # 50-50 do rotation or not
        print("adding rotation option for imgs_tf: %d" % config.rot_val)
        if config.always_rot:
            print("always_rot")
            tf2_list += [torchvision.transforms.RandomRotation(config.rot_val)]
        else:
            print("not always_rot")
            tf2_list += [torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

    if config.crop_other:
        imgs_tf_crops = []
        for tf2_crop_sz in config.tf2_crop_szs:
            if config.tf2_crop == "random":
                print("selected random crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomCrop(tf2_crop_sz)
            elif config.tf2_crop == "centre_half":
                print("selected centre_half crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomCrop(tf2_crop_sz),
                    torchvision.transforms.CenterCrop(tf2_crop_sz)
                ])
            elif config.tf2_crop == "centre":
                print("selected centre crop for tf2")
                tf2_crop_fn = torchvision.transforms.CenterCrop(tf2_crop_sz)
            else:
                assert False

            print("adding crop size option for imgs_tf: %d" % tf2_crop_sz)
            imgs_tf_crops.append(tf2_crop_fn)

        tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]

    tf2_list += [torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                               config.input_sz])))]

    if not config.no_flip:
        print("adding flip")
        tf2_list += [torchvision.transforms.RandomHorizontalFlip()]
    else:
        print("not adding flip")

    if not config.no_jitter:
        print("adding jitter")
        tf2_list += [
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                               saturation=0.4, hue=0.125)]
    else:
        print("not adding jitter")

    tf2_list += [torchvision.transforms.ToTensor()]

    # admin transforms
    if config.demean:
        print("demeaning data")
        tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
    else:
        print("not demeaning data")

    if config.per_img_demean:
        print("per image demeaning data")
        tf1_list.append(per_img_demean)
        tf2_list.append(per_img_demean)
        tf3_list.append(per_img_demean)
    else:
        print("not per image demeaning data")

    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)

    return tf1, tf2, tf3


# Used by sobel and greyscale clustering twohead scripts -----------------------

def cluster_twohead_create_dataloaders(config):
    assert (config.mode == "IID")
    assert config.twohead

    target_transform = None

    if "CIFAR" in config.dataset:
        config.train_partitions_head_A = [True, False]
        config.train_partitions_head_B = config.train_partitions_head_A

        config.mapping_assignment_partitions = [True, False]
        config.mapping_test_partitions = [True, False]

        if config.dataset == "CIFAR10":
            dataset_class = torchvision.datasets.CIFAR10
        elif config.dataset == "CIFAR100":
            dataset_class = torchvision.datasets.CIFAR100
        elif config.dataset == "CIFAR20":
            dataset_class = torchvision.datasets.CIFAR100
            target_transform = _cifar100_to_cifar20
        else:
            assert False

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "STL10":
        assert config.mix_train
        if not config.stl_leave_out_unlabelled:
            print("adding unlabelled data for STL10")
            config.train_partitions_head_A = ["train+unlabeled", "test"]
        else:
            print("not using unlabelled data for STL10")
            config.train_partitions_head_A = ["train", "test"]

        config.train_partitions_head_B = ["train", "test"]

        config.mapping_assignment_partitions = ["train", "test"]
        config.mapping_test_partitions = ["train", "test"]

        dataset_class = torchvision.datasets.STL10

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "MNIST":
        config.train_partitions_head_A = [True, False]
        config.train_partitions_head_B = config.train_partitions_head_A

        config.mapping_assignment_partitions = [True, False]
        config.mapping_test_partitions = [True, False]

        dataset_class = torchvision.datasets.MNIST

        tf1, tf2, tf3 = greyscale_make_transforms(config)

    else:
        assert False

    print("Making datasets with %s and %s" % (dataset_class, target_transform))
    sys.stdout.flush()

    dataloaders_head_a = \
        _create_dataloaders(config, dataset_class, tf1, tf2,
                            partitions=config.train_partitions_head_A,
                            target_transform=target_transform)

    dataloaders_head_b = \
        _create_dataloaders(config, dataset_class, tf1, tf2,
                            partitions=config.train_partitions_head_B,
                            target_transform=target_transform)

    mapping_assignment_dataloader = \
        _create_mapping_loader(config, dataset_class, tf3,
                               partitions=config.mapping_assignment_partitions,
                               target_transform=target_transform)

    mapping_test_dataloader = \
        _create_mapping_loader(config, dataset_class, tf3,
                               partitions=config.mapping_test_partitions,
                               target_transform=target_transform)

    return dataloaders_head_a, dataloaders_head_b, \
        mapping_assignment_dataloader, mapping_test_dataloader


# Used by sobel and greyscale clustering single head scripts -------------------

def cluster_create_dataloaders(config):
    assert (config.mode == "IID+")
    assert (not config.twohead)

    target_transform = None

    # separate train/test sets
    if "CIFAR" in config.dataset:
        config.train_partitions = [True]
        config.mapping_assignment_partitions = [True]
        config.mapping_test_partitions = [False]

        if config.dataset == "CIFAR10":
            dataset_class = torchvision.datasets.CIFAR10
        elif config.dataset == "CIFAR100":
            dataset_class = torchvision.datasets.CIFAR100
        elif config.dataset == "CIFAR20":
            dataset_class = torchvision.datasets.CIFAR100
            target_transform = _cifar100_to_cifar20
        else:
            assert False

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "STL10":
        config.train_partitions = ["train+unlabeled"]
        config.mapping_assignment_partitions = ["train"]
        config.mapping_test_partitions = ["test"]

        dataset_class = torchvision.datasets.STL10

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(config)

    elif config.dataset == "MNIST":
        config.train_partitions = [True]
        config.mapping_assignment_partitions = [True]
        config.mapping_test_partitions = [False]

        dataset_class = torchvision.datasets.MNIST

        tf1, tf2, tf3 = greyscale_make_transforms(config)

    else:
        assert False

    print("Making datasets with %s and %s" % (dataset_class, target_transform))
    sys.stdout.flush()

    dataloaders = \
        _create_dataloaders(config, dataset_class, tf1, tf2,
                            partitions=config.train_partitions,
                            target_transform=target_transform)

    mapping_assignment_dataloader = \
        _create_mapping_loader(config, dataset_class, tf3,
                               partitions=config.mapping_assignment_partitions,
                               target_transform=target_transform)

    mapping_test_dataloader = \
        _create_mapping_loader(config, dataset_class, tf3,
                               partitions=config.mapping_test_partitions,
                               target_transform=target_transform)

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


# Other generic data creation functions ----------------------------------------

def make_stl_data(config, tf1=None, tf2=None, tf3=None,
                  truncate_assign=False, truncate_pc=None):
    assert (tf3 is not None)
    if (tf1 is not None) and (tf2 is not None):
        dataloaders = _create_dataloaders(config, torchvision.datasets.STL10, tf1,
                                          tf2,
                                          partitions=config.train_partitions_head_B)

    mapping_assignment_dataloader = _create_mapping_loader(
        config, torchvision.datasets.STL10, tf3,
        partitions=config.mapping_assignment_partitions,
        truncate=truncate_assign, truncate_pc=truncate_pc)

    mapping_test_dataloader = _create_mapping_loader(
        config, torchvision.datasets.STL10, tf3,
        partitions=config.mapping_test_partitions)

    if (tf1 is not None) and (tf2 is not None):
        return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader
    else:
        return mapping_assignment_dataloader, mapping_test_dataloader


def make_cifar_data(config, tf1=None, tf2=None, tf3=None,
                    truncate_assign=False, truncate_pc=None):
    target_transform = None

    if config.dataset == "CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
    elif config.dataset == "CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
    elif config.dataset == "CIFAR20":
        dataset_class = torchvision.datasets.CIFAR100
        target_transform = _cifar100_to_cifar20
    else:
        assert False

    assert (tf3 is not None)
    if (tf1 is not None) and (tf2 is not None):
        dataloaders = _create_dataloaders(config, dataset_class, tf1, tf2,
                                          partitions=config.train_partitions_head_B,
                                          target_transform=target_transform)

    mapping_assignment_dataloader = _create_mapping_loader(
        config, dataset_class, tf3, config.mapping_assignment_partitions,
        target_transform=target_transform,
        truncate=truncate_assign, truncate_pc=truncate_pc)

    mapping_test_dataloader = _create_mapping_loader(
        config, dataset_class, tf3, config.mapping_test_partitions,
        target_transform=target_transform)

    if (tf1 is not None) and (tf2 is not None):
        return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader
    else:
        return mapping_assignment_dataloader, mapping_test_dataloader


def make_mnist_data(config, tf1=None, tf2=None, tf3=None,
                    truncate_assign=False, truncate_pc=None):
    assert (tf3 is not None)
    if (tf1 is not None) and (tf2 is not None):
        dataloaders = _create_dataloaders(config, torchvision.datasets.MNIST, tf1,
                                          tf2,
                                          partitions=config.train_partitions_head_B)

    mapping_assignment_dataloader = _create_mapping_loader(
        config, torchvision.datasets.MNIST, tf3,
        config.mapping_assignment_partitions,
        truncate=truncate_assign, truncate_pc=truncate_pc)

    mapping_test_dataloader = _create_mapping_loader(
        config, torchvision.datasets.MNIST, tf3,
        config.mapping_test_partitions)

    if (tf1 is not None) and (tf2 is not None):
        return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader
    else:
        return mapping_assignment_dataloader, mapping_test_dataloader


# Data creation helpers --------------------------------------------------------

def _create_dataloaders(config, dataset_class, tf1, tf2,
                        partitions,
                        target_transform=None,
                        shuffle=False,
                        sampler_=None):
    train_imgs_list = []
    for train_partition in partitions:
        if "STL10" == config.dataset:
            train_imgs_curr = dataset_class(
                root=config.dataset_root,
                transform=tf1,
                split=train_partition,
                target_transform=target_transform, download=True)
        else:
            train_imgs_curr = dataset_class(
                root=config.dataset_root,
                transform=tf1,
                train=train_partition,
                target_transform=target_transform, download=True)

        if hasattr(config, "mix_train"):
            if config.mix_train and (train_partition == "train+unlabeled"):
                train_imgs_curr = reorder_train_deterministic(train_imgs_curr)
        train_imgs_list.append(train_imgs_curr)

    train_imgs = ConcatDataset(train_imgs_list)
    train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=shuffle,
                                                   num_workers=0,
                                                   drop_last=False, sampler=sampler_)

    if not shuffle:
        assert (isinstance(train_dataloader.sampler,
                           torch.utils.data.sampler.SequentialSampler))
    dataloaders = [train_dataloader]

    for d_i in range(config.num_dataloaders):
        print("Creating auxiliary dataloader ind %d out of %d time %s" %
              (d_i, config.num_dataloaders, datetime.now()))
        sys.stdout.flush()

        train_tf_imgs_list = []
        for train_partition in partitions:
            if "STL10" == config.dataset:
                train_imgs_tf_curr = dataset_class(
                    root=config.dataset_root,
                    transform=tf2,  # random per call
                    split=train_partition,
                    target_transform=target_transform)
            else:
                train_imgs_tf_curr = dataset_class(
                    root=config.dataset_root,
                    transform=tf2,
                    train=train_partition,
                    target_transform=target_transform)

            if hasattr(config, "mix_train"):
                if config.mix_train and (train_partition == "train+unlabeled"):
                    train_imgs_tf_curr = reorder_train_deterministic(train_imgs_tf_curr)
            train_tf_imgs_list.append(train_imgs_tf_curr)

        train_imgs_tf = ConcatDataset(train_tf_imgs_list)
        train_tf_dataloader = \
            torch.utils.data.DataLoader(train_imgs_tf,
                                        batch_size=config.dataloader_batch_sz,
                                        shuffle=shuffle,
                                        num_workers=0,
                                        drop_last=False, sampler=sampler_)

        if not shuffle:
            assert (isinstance(train_tf_dataloader.sampler,
                               torch.utils.data.sampler.SequentialSampler))
        assert (len(train_dataloader) == len(train_tf_dataloader))
        dataloaders.append(train_tf_dataloader)

    num_train_batches = len(dataloaders[0])
    print("Length of datasets vector %d" % len(dataloaders))
    print("Number of batches per epoch: %d" % num_train_batches)
    sys.stdout.flush()

    return dataloaders


def _create_mapping_loader(config, dataset_class, tf3, partitions,
                           target_transform=None,
                           truncate=False, truncate_pc=None,
                           tencrop=False,
                           shuffle=False, sampler_=None):
    if truncate:
        print("Note: creating mapping loader with truncate == True")

    if tencrop:
        assert (tf3 is None)

    imgs_list = []
    for partition in partitions:
        if "STL10" == config.dataset:
            imgs_curr = dataset_class(
                root=config.dataset_root,
                transform=tf3,
                split=partition,
                target_transform=target_transform)
        else:
            imgs_curr = dataset_class(
                root=config.dataset_root,
                transform=tf3,
                train=partition,
                target_transform=target_transform)

        if truncate:
            print("shrinking dataset from %d" % len(imgs_curr))
            imgs_curr = TruncatedDataset(imgs_curr, pc=truncate_pc)
            print("... to %d" % len(imgs_curr))

        # if tencrop:
        #   imgs_curr = TenCropAndFinish(imgs_curr, input_sz=config.input_sz,
        #                                include_rgb=config.include_rgb)

        imgs_list.append(imgs_curr)

    imgs = ConcatDataset(imgs_list)
    dataloader = torch.utils.data.DataLoader(imgs,
                                             batch_size=config.batch_sz,
                                             # full batch
                                             shuffle=shuffle,
                                             num_workers=0,
                                             drop_last=False, sampler=sampler_)

    if not shuffle:
        assert (isinstance(dataloader.sampler,
                           torch.utils.data.sampler.SequentialSampler))
    return dataloader


def _cifar100_to_cifar20(target):
    # obtained from cifar_test script
    _dict = \
        {0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13}

    return _dict[target]


def _clustering_get_data(config, net, dataloader, sobel=False,
                         using_ir=False, get_soft=False, verbose=0):
    """
    Returns cuda tensors for flat preds and targets.
    """

    assert (not using_ir)  # sanity; IR used by segmentation only

    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda()
    flat_predss_all = [torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda() for _ in
                       range(config.num_sub_heads)]

    if get_soft:
        soft_predss_all = [torch.zeros((num_batches * config.batch_sz,
                                        config.output_k),
                                       dtype=torch.float32).cuda() for _ in range(
            config.num_sub_heads)]

    num_test = 0
    for b_i, batch in enumerate(dataloader):
        # batch = [tensor1, tensor2], list of two tensors
        # batch[0]: tensor1.shape = [660, 1, 32, 32] for cifar data
        # batch[1]: tensor2.shape = [660]

        # plt.imshow(torch.squeeze(batch[0][0].permute(1, 2, 0), 2), cmap='gray')
        # plt.show()
        # print(batch[0][0])
        # print('b_i, batch size ', b_i, batch[0].shape, batch[1].shape)

        imgs = batch[0].cuda()

        if sobel:
            imgs = sobel_process(imgs, config.include_rgb, using_ir=using_ir)

        flat_targets = batch[1]

        with torch.no_grad():
            x_outs = net(imgs)

        assert (x_outs[0].shape[1] == config.output_k)
        assert (len(x_outs[0].shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        for i in range(config.num_sub_heads):
            x_outs_curr = x_outs[i]
            flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
            flat_predss_all[i][start_i:(start_i + num_test_curr)] = flat_preds_curr

            if get_soft:
                soft_predss_all[i][start_i:(start_i + num_test_curr), :] = x_outs_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_predss_all = [flat_predss_all[i][:num_test] for i in
                       range(config.num_sub_heads)]
    flat_targets_all = flat_targets_all[:num_test]

    if not get_soft:
        return flat_predss_all, flat_targets_all
    else:
        soft_predss_all = [soft_predss_all[i][:num_test] for i in
                           range(config.num_sub_heads)]

        return flat_predss_all, flat_targets_all, soft_predss_all


def cluster_subheads_eval(config, net,
                          mapping_assignment_dataloader,
                          mapping_test_dataloader,
                          sobel,
                          using_ir=False,
                          get_data_fn=_clustering_get_data,
                          use_sub_head=None,
                          verbose=0):
    """
    Used by both clustering and segmentation.
    Returns metrics for test set.
    Get result from average accuracy of all sub_heads (mean and std).
    All matches are made from training data.
    Best head metric, which is order selective unlike mean/std, is taken from
    best head determined by training data (but metric computed on test data).

    ^ detail only matters for IID+/semisup where there's a train/test split.

    Option to choose best sub_head either based on loss (set use_head in main
    script), or eval. Former does not use labels for the selection at all and this
    has negligible impact on accuracy metric for our models.
    """

    all_matches, train_accs = _get_assignment_data_matches(net,
                                                           mapping_assignment_dataloader,
                                                           config,
                                                           sobel=sobel,
                                                           using_ir=using_ir,
                                                           get_data_fn=get_data_fn,
                                                           verbose=verbose)

    best_sub_head_eval = np.argmax(train_accs)
    if (config.num_sub_heads > 1) and (use_sub_head is not None):
        best_sub_head = use_sub_head
    else:
        best_sub_head = best_sub_head_eval

    if config.mode == "IID":
        assert (
            config.mapping_assignment_partitions == config.mapping_test_partitions)
        test_accs = train_accs
    elif config.mode == "IID+":
        flat_predss_all, flat_targets_all, = \
            get_data_fn(config, net, mapping_test_dataloader, sobel=sobel,
                        using_ir=using_ir,
                        verbose=verbose)

        num_samples = flat_targets_all.shape[0]
        test_accs = np.zeros(config.num_sub_heads, dtype=np.float32)
        for i in range(config.num_sub_heads):
            reordered_preds = torch.zeros(num_samples,
                                          dtype=flat_predss_all[0].dtype).cuda()
            for pred_i, target_i in all_matches[i]:
                reordered_preds[flat_predss_all[i] == pred_i] = target_i
            test_acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose=0)

            test_accs[i] = test_acc
    else:
        assert False

    return {"test_accs": list(test_accs),
            "avg": np.mean(test_accs),
            "std": np.std(test_accs),
            "best": test_accs[best_sub_head],
            "worst": test_accs.min(),
            "best_train_sub_head": best_sub_head,  # from training data
            "best_train_sub_head_match": all_matches[best_sub_head],
            "train_accs": list(train_accs)}


def _get_assignment_data_matches(net, mapping_assignment_dataloader, config,
                                 sobel=False,
                                 using_ir=False,
                                 get_data_fn=None,
                                 just_matches=False,
                                 verbose=0):
    """
    Get all best matches per head based on train set i.e. mapping_assign,
    and mapping_assign accs.
    """

    if verbose:
        print("calling cluster eval direct (helper) %s" % datetime.now())
        sys.stdout.flush()

    flat_predss_all, flat_targets_all = \
        get_data_fn(config, net, mapping_assignment_dataloader, sobel=sobel,
                    using_ir=using_ir,
                    verbose=verbose)

    if verbose:
        print("getting data fn has completed %s" % datetime.now())
        print("flat_targets_all %s, flat_predss_all[0] %s" %
              (list(flat_targets_all.shape), list(flat_predss_all[0].shape)))
        sys.stdout.flush()

    num_test = flat_targets_all.shape[0]
    if verbose == 2:
        print("num_test: %d" % num_test)
        for c in range(config.gt_k):
            print("gt_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

    assert (flat_predss_all[0].shape == flat_targets_all.shape)
    num_samples = flat_targets_all.shape[0]

    all_matches = []
    if not just_matches:
        all_accs = np.zeros(config.num_sub_heads, dtype=np.float32)

    for i in range(config.num_sub_heads):
        if verbose:
            print("starting head %d with eval mode %s, %s" % (i, config.eval_mode,
                                                              datetime.now()))
            sys.stdout.flush()

        if config.eval_mode == "hung":
            match = _hungarian_match(flat_predss_all[i], flat_targets_all,
                                     preds_k=config.output_k,
                                     targets_k=config.gt_k)
        elif config.eval_mode == "orig":
            match = _original_match(flat_predss_all[i], flat_targets_all,
                                    preds_k=config.output_k,
                                    targets_k=config.gt_k)
        else:
            assert False

        if verbose:
            print("got match %s" % (datetime.now()))
            sys.stdout.flush()

        all_matches.append(match)

        if not just_matches:
            # reorder predictions to be same cluster assignments as gt_k
            found = torch.zeros(config.output_k)
            reordered_preds = torch.zeros(num_samples,
                                          dtype=flat_predss_all[0].dtype).cuda()

            for pred_i, target_i in match:
                reordered_preds[flat_predss_all[i] == int(pred_i)] = int(target_i)
                found[pred_i] = 1
                if verbose == 2:
                    print((pred_i, target_i))
            assert (found.sum() == config.output_k)  # each output_k must get mapped

            if verbose:
                print("reordered %s" % (datetime.now()))
                sys.stdout.flush()

            print('subhead ', i)
            print('reordered ', reordered_preds)
            print('targets', flat_targets_all)
            acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose)
            all_accs[i] = acc

    if just_matches:
        return all_matches
    else:
        return all_matches, all_accs


def get_subhead_using_loss(config, dataloaders_head_b, net, sobel, lamb,
                           compare=False):
    net.eval()

    head = "B"  # main output head
    dataloaders = dataloaders_head_b
    iterators = (d for d in dataloaders)

    b_i = 0
    loss_per_sub_head = np.zeros(config.num_sub_heads)
    # for tup in itertools.izip(*iterators):
    for tup in zip(*iterators):
        net.module.zero_grad()

        dim = config.in_channels
        if sobel:
            dim -= 1

        all_imgs = torch.zeros(config.batch_sz, dim,
                               config.input_sz,
                               config.input_sz).cuda()
        all_imgs_tf = torch.zeros(config.batch_sz, dim,
                                  config.input_sz,
                                  config.input_sz).cuda()

        imgs_curr = tup[0][0]  # always the first
        curr_batch_sz = imgs_curr.size(0)
        for d_i in range(config.num_dataloaders):
            imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
            assert (curr_batch_sz == imgs_tf_curr.size(0))

            actual_batch_start = d_i * curr_batch_sz
            actual_batch_end = actual_batch_start + curr_batch_sz
            all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
                imgs_curr.cuda()
            all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
                imgs_tf_curr.cuda()

        curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
        all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
        all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

        if sobel:
            all_imgs = sobel_process(all_imgs, config.include_rgb)
            all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

        with torch.no_grad():
            x_outs = net(all_imgs, head=head)
            x_tf_outs = net(all_imgs_tf, head=head)

        for i in range(config.num_sub_heads):
            loss, loss_no_lamb, _ = iid_loss(x_outs[i], x_tf_outs[i],
                                          lamb=lamb)
            loss_per_sub_head[i] += loss.item()

        if b_i % 100 == 0:
            print("at batch %d" % b_i)
            sys.stdout.flush()
        b_i += 1

    best_sub_head_loss = np.argmin(loss_per_sub_head)

    if compare:
        print(loss_per_sub_head)
        print("best sub_head by loss: %d" % best_sub_head_loss)

        best_epoch = np.argmax(np.array(config.epoch_acc))
        if "best_train_sub_head" in config.epoch_stats[best_epoch]:
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_train_sub_head"]
            test_accs = config.epoch_stats[best_epoch]["test_accs"]
        else:  # older config version
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_head"]
            test_accs = config.epoch_stats[best_epoch]["all"]

        print("best sub_head by eval: %d" % best_sub_head_eval)

        print("... loss select acc: %f, eval select acc: %f" %
              (test_accs[best_sub_head_loss],
               test_accs[best_sub_head_eval]))

    net.train()

    return best_sub_head_loss


def cluster_eval(config, net, mapping_assignment_dataloader,
                 mapping_test_dataloader, sobel,
                 use_sub_head=None, print_stats=False):
    if config.double_eval:
        # Pytorch's behaviour varies depending on whether .eval() is called or not
        # The effect is batchnorm updates if .eval() is not called
        # So double eval can be used (optionally) for IID, where train = test set.
        # https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm2d

        stats_dict2 = cluster_subheads_eval(config, net,
                                            mapping_assignment_dataloader=mapping_assignment_dataloader,
                                            mapping_test_dataloader=mapping_test_dataloader,
                                            sobel=sobel,
                                            use_sub_head=use_sub_head)

        if print_stats:
            print("double eval stats:")
            print(stats_dict2)
        else:
            config.double_eval_stats.append(stats_dict2)
            config.double_eval_acc.append(stats_dict2["best"])
            config.double_eval_avg_subhead_acc.append(stats_dict2["avg"])

    net.eval()
    stats_dict = cluster_subheads_eval(config, net,
                                       mapping_assignment_dataloader=mapping_assignment_dataloader,
                                       mapping_test_dataloader=mapping_test_dataloader,
                                       sobel=sobel,
                                       use_sub_head=use_sub_head)
    net.train()

    if print_stats:
        print("eval stats:")
        print(stats_dict)                   # ?? no return??
        return False
    else:
        acc = stats_dict["best"]
        is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

        config.epoch_stats.append(stats_dict)
        config.epoch_acc.append(acc)
        config.epoch_avg_subhead_acc.append(stats_dict["avg"])

        return is_best


def multioutput_k_means_assess(config, x_outs_all, targets, verbose=0):
    # assert (False)  # outdated function
    num_sub_heads = len(x_outs_all)
    print("assessing multioutput using k-means, heads: %d" % num_sub_heads)

    accs = []
    nmis = []
    aris = []
    best_i = None
    for i in range(num_sub_heads):
        x_outs = x_outs_all[i]  # not flat
        n, dlen = x_outs.shape
        # random_state=0
        kmeans = KMeans(n_clusters=config.gt_k).fit(x_outs)

        n2, = targets.shape
        assert (n == n2)
        assert (max(targets) == (config.gt_k - 1))

        flat_predictions = kmeans.labels_

        # get into same indices

        if config.kmeans_map == "many_to_one":
            assert (config.eval_mode == "orig")
            match = _original_match(flat_predictions, targets,
                                    preds_k=config.output_k,
                                    targets_k=config.gt_k)
        elif config.kmeans_map == "one_to_one":
            # hungarian
            match = _hungarian_match(flat_predictions, targets,
                                     preds_k=config.output_k,
                                     targets_k=config.gt_k)
        else:
            assert False

        # reorder predictions to be same cluster assignments as gt_k
        # reordered_preds = np.zeros(n)
        reordered_preds = torch.zeros(n)
        for pred_i, target_i in match:
            reordered_preds[flat_predictions == pred_i] = target_i
            if verbose > 1:
                print((pred_i, target_i))

        acc = _acc(reordered_preds, targets, config.gt_k, verbose)

        # this works because for max acc, will get set and never un-set
        if (best_i is None) or (acc > max(accs)):
            best_i = i

        if verbose > 0:
            print("head %d acc %f" % (i, acc))

        accs.append(acc)

    return accs[best_i], nmis[best_i], aris[best_i]


# for all heads/models, keep the colouring consistent
GT_TO_ORDER = [2, 5, 3, 8, 6, 7, 0, 9, 1, 4]


def save_progress(config, net, mapping_assignment_dataloader,
                  mapping_test_dataloader, index, sobel, render_count):
    """
    Draws all predictions using convex combination.
    """

    # Using this for MNIST
    if sobel:
        raise NotImplementedError

    prog_out_dir = os.path.join(config.out_dir, "progression")
    if not os.path.exists(prog_out_dir):
        os.makedirs(prog_out_dir)

    # find the best head
    using_ir = False  # whole images
    all_matches, train_accs = _get_assignment_data_matches(net,
                                                           mapping_assignment_dataloader,
                                                           config,
                                                           sobel=sobel,
                                                           using_ir=using_ir,
                                                           get_data_fn=_clustering_get_data)

    best_sub_head = np.argmax(train_accs)
    match = all_matches[best_sub_head]

    # get clustering results
    flat_predss_all, flat_targets_all, soft_predss_all = \
        _clustering_get_data(config, net, mapping_test_dataloader, sobel=sobel,
                             using_ir=using_ir, get_soft=True)
    soft_preds = soft_predss_all[best_sub_head]

    num_samples, cc = soft_preds.shape
    assert (cc == config.gt_k)
    reordered_soft_preds = torch.zeros((num_samples, config.gt_k),
                                       dtype=soft_preds.dtype).cuda()
    for pred_i, target_i in match:
        reordered_soft_preds[:, GT_TO_ORDER[target_i]] += \
            soft_preds[:, pred_i]  # 1-1 for IIC
    reordered_soft_preds = reordered_soft_preds.cpu().numpy()

    # render point cloud in GT order ---------------------------------------------
    hues = torch.linspace(0.0, 1.0, config.gt_k + 1)[0:-1]  # ignore last one
    best_colours = [list((np.array(hsv_to_rgb(hue, 0.8, 0.8)) * 255.).astype(
        np.uint8)) for hue in hues]

    all_colours = [best_colours]

    for colour_i, colours in enumerate(all_colours):
        scale = 50  # [-1, 1] -> [-scale, scale]
        border = 24  # averages are in the borders
        point_half_side = 1  # size 2 * pixel_half_side + 1

        half_border = int(border * 0.5)

        average_half_side = int(half_border * np.cos(np.radians(45)))
        average_side = average_half_side * 2

        image = np.ones((2 * (scale + border), 2 * (scale + border), 3),
                        dtype=np.uint8) * 255

        # image = np.zeros((2 * (scale + border), 2 * (scale + border), 3),
        #                dtype=np.int32)

        for i in range(num_samples):
            # in range [-1, 1] -> [0, 2 * scale] -> [border, 2 * scale + border]
            coord = get_coord(reordered_soft_preds[i, :], num_classes=config.gt_k)
            coord = (coord * scale + scale).astype(np.int32)
            coord += border
            pt_start = coord - point_half_side
            pt_end = coord + point_half_side

            render_c = GT_TO_ORDER[flat_targets_all[i]]
            colour = (np.array(colours[render_c])).astype(np.uint8)
            image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = np.reshape(
                colour, (1, 1, 3))

        # add on the average image per cluster in the border
        # -------------------------
        # dataloaders not shuffled, or jittered here
        averaged_imgs = [np.zeros((config.input_sz, config.input_sz, 1)) for _ in
                         range(config.gt_k)]
        averaged_imgs_norm = [0. for _ in range(config.gt_k)]
        counter = 0
        for b_i, batch in enumerate(mapping_test_dataloader):
            imgs = batch[0].numpy()  # n, c, h, w
            n, c, h, w = imgs.shape
            assert (c == 1)

            for offset in range(n):
                img_i = counter + offset
                img = imgs[offset]
                img = img.transpose((1, 2, 0))
                img = img * 255

                # already in right order
                predicted_cluster_render = torch.argmax(reordered_soft_preds[img_i, :])
                predicted_cluster_gt_weight = reordered_soft_preds[
                    img_i, predicted_cluster_render]

                averaged_imgs[predicted_cluster_render.item()] += \
                    predicted_cluster_gt_weight * img
                averaged_imgs_norm[predicted_cluster_render.item()] += \
                    predicted_cluster_gt_weight

            counter += n

        for c in range(config.gt_k):
            if averaged_imgs_norm[c] > sys.float_info.epsilon:
                averaged_imgs[c] /= averaged_imgs_norm[c]

            averaged_img = averaged_imgs[c].astype(np.uint8)
            averaged_img = averaged_img.repeat(3, axis=2)
            averaged_img = Image.fromarray(averaged_img)
            averaged_img = averaged_img.resize((average_side, average_side),
                                               Image.BILINEAR)
            averaged_img = np.array(averaged_img)

            coord = np.zeros(config.gt_k)
            coord[c] = 1.
            coord = get_coord(coord, num_classes=config.gt_k)

            # recall coord is for center of image
            # [-1, 1] -> [0, 2 * (scale + half_border)]
            coord = (coord * (scale + half_border) + (scale + half_border)).astype(
                np.int32)
            # -> [half_border, 2 * (scale + half_border) + half_border]
            coord += half_border

            pt_start = coord - average_half_side
            pt_end = coord + average_half_side  # exclusive

            image[pt_start[0]:pt_end[0], pt_start[1]:pt_end[1], :] = averaged_img

        # save to out_dir ---------------------------
        img = Image.fromarray(image)
        img.save(os.path.join(prog_out_dir,
                              "%d_run_%d_colour_%d_pointcloud_%s.png" %
                              (config.model_ind, render_count, colour_i, index)))


def get_coord(probs, num_classes):
    # computes coordinate for 1 sample based on probability distribution over c
    coords_total = np.zeros(2, dtype=np.float32)
    probs_sum = probs.sum()

    fst_angle = 0.

    for c in range(num_classes):
        # compute x, y coordinates
        coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle
        coords[0] = np.sin(coords[0])
        coords[1] = np.cos(coords[1])
        coords_total += (probs[c] / probs_sum) * coords
    return coords_total
