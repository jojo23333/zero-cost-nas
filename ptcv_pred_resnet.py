# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import argparse
import os

from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.model_provider import _models as ptcv_models
from ptcv_nets import ptcv_accs_cf10, ptcv_accs_cf100, ptcv_accs_svhn, ptcv_accs_imgnet
from foresight.pruners import *
from foresight.dataset import *
from torchvision.models.resnet import _resnet, BasicBlock

def build_resnet(d3=2, c=16):
    return _resnet(BasicBlock, [3, 4, d3, 3], num_classes=10, inplanes=c, stem_stride=1, weights=None, progress=False)

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if (args.dataset == 'cifar10' or args.dataset=='svhn') else 1000 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for PTCV')
    parser.add_argument('--outdir', default='.', type=str, help='output directory')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--datadir', type=str, default='_dataset', help='data location')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()

    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'cifar10':
        ptcv_accs = ptcv_accs_cf10
    elif args.dataset == 'cifar100':
        ptcv_accs = ptcv_accs_cf100
    elif args.dataset == 'svhn':
        ptcv_accs = ptcv_accs_svhn
    elif args.dataset == 'ImageNet1k':
        ptcv_accs = ptcv_accs_imgnet
    else:
        raise NotImplementedError

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, datadir=args.datadir)

    fn = f'pred_ptcv_{args.dataset}'+('_pretrain' if args.pretrain else '')+'.p'

    print(f'Saving to = {args.outdir}, {fn}')

    all_res = []

    depth = range(1, 24)
    width = range(1, 32)
    for d in depth:
        for w in width:
            res = {'name': f'{d}_{w}'}

            print(f'Working on {d}_{w}..')
            net = build_resnet(d3=d, c=w)

            try:
                net.to(args.device)
                measures = predictive.find_measures(net, 
                                            train_loader, 
                                            (args.dataload, args.dataload_info, get_num_classes(args)),
                                            args.device)
            except Exception as e:
                del net
                torch.cuda.empty_cache()
                print(e)
                print('continue')
                continue

            res['measures']= measures
            
            all_res.append(res)
            torch.save(all_res, './resnet.pth')

    # pf=open(fn, 'wb')
    # pickle.dump(all_res, pf)
    # pf.close()

    src = fn
    dst = os.path.join(args.outdir, fn)
    from shutil import copyfile
    copyfile(src, dst)