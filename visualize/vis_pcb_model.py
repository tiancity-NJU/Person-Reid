from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0,osp.abspath(osp.dirname(__file__)+osp.sep+'..'))

from reid.utils import to_numpy
from reid import datasets
from reid import models
from reid.models import pcb
from reid.dist_metric import DistanceMetric
from reid.evaluators import Evaluator


from reid.evaluators import extract_pcb_features
from reid.evaluators import pairwise_distance

from reid.utils.visualize import get_rank_list
from reid.utils.visualize import save_rank_list_to_im
from reid.utils.visualize import get_frame
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint




def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)


    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])


    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset , test_loader



def get_distmat(model,data_loader,query,gallery,metric,filter = False):
    """

    :param model:   reid model
    :param data_loader:  test_loader
    :param query:  list of tuple    [('xxxxx.jpg',id,cam)]
    :param gallery:   list of tuple [('xxxxx.jpg',id,cam)]
    :param metric:  default euclidean distance
    :return:
        distmat: query*gallery mat (single query)
        query:   array like   [['xxxxx.jpg','id','cam']...]
        gallery: array like   [['xxxxx.jpg','id','cam']...]
    """
    features, _ = extract_pcb_features(model, data_loader)
    distmat = to_numpy(pairwise_distance(features, query, gallery, metric=metric))
    # indices=torch.LongTensor([0,3,5,6,7])
    # y=torch.index_select(distmat,0,indices)

    if filter:
        distmat = np.array([distmat[i] for i in range(len(query)) if
                        i == 0 or query[i][1] != query[i - 1][1]])  # keep one image each person (for query)
        query = np.array([query[i] for i in range(len(query)) if i == 0 or query[i][1] != query[i - 1][1]])


    return distmat,query,gallery





def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True


    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (384, 128)

    dataset, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval)


    model = pcb.PCB_resnet(args.arch,num_stripes=args.num_stripes,local_conv_out_channel=args.features,num_class=702)


    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        pretrained_dict=checkpoint['state_dict']
        model_dict=model.state_dict()
        pretrained_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    else: raise RuntimeError('please load a pretrained model!')

    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    print("Test:")
    # evaluator.evaluate_pcb(test_loader, dataset.query, dataset.gallery, metric)

    distmat,query,gallery = get_distmat(model,test_loader,dataset.query,dataset.gallery,metric)

    q_frames = np.array([get_frame(img) for img, _, _ in query])
    g_frames = np.array([get_frame(img) for img, _, _ in gallery])

    q_im_paths = np.array([osp.join(args.data_dir, args.dataset, 'images', img) for img, _, _ in query])
    q_ids = np.array([int(id) for _, id, _ in query])
    q_cams = np.array([int(c) for _, _, c in query])

    g_im_paths = np.array([osp.join(args.data_dir, args.dataset, 'images', img) for img, _, _ in gallery])
    g_ids = np.array([int(id) for _, id, _ in gallery])
    g_cams = np.array([int(c) for _, _, c in gallery])

    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_paths=np.array([args.save_dir for _ , _ , _ in query])

    count=0
    for dist_vec, q_id, q_cam, q_im_path, q_frame, save_path in zip(distmat, q_ids, q_cams, q_im_paths, q_frames, save_paths):
        count+=1
        rank_list, same_id = get_rank_list(
            dist_vec, q_id, q_cam, q_frame, g_ids, g_cams, g_frames, args.rank_list_size,temporal=args.temporal)

        save_rank_list_to_im(rank_list, same_id, q_im_path, g_im_paths, save_path,ignore_id=True)
        if count%50 == 0:
            print('{:d} is finished '.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='test',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-r', '--rank_list_size', type=int, default=5)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception",default=384)
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception",default=128)
    parser.add_argument('--combine_trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--num_stripes', type=int, default=6)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.abspath(osp.dirname(__file__)+osp.sep+'..')

    parser.add_argument('--save-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'vis_space_dukemtmc'))

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    parser.add_argument('--temporal',action='store_true',default=False)   # whether use easy temporal info
    main(parser.parse_args())

