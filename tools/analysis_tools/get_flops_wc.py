# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector
import re


try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    ori_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    from ptflops import get_model_complexity_info
    num_all =0
    for name, p in model.named_parameters():
        num = 1
        for n in p.shape:
            num =num*n
        num_all +=num
        print('%s: %f M'%(name,num/1000000))
    print('all: %f M' % ( num_all / 1000000))

    macs, params = get_model_complexity_info(model, input_shape, as_strings=False,  print_per_layer_stat=True, verbose=True)
    flops = macs* 2
    print('Params all: %f M' % ( num_all / 1000000))

    print('FLOPs all: %f G' % ( flops / 1000000000))

    pass


if __name__ == '__main__':
    main()
