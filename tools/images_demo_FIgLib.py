# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
from argparse import ArgumentParser
import numpy as np
import tqdm,cv2

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    img_list = os.listdir(args.img_dir)
    img_list = [os.path.join(args.img_dir,v) for v in img_list if (v.endswith('.jpg') or v.endswith('.png'))]
    results_all = []

    for img in tqdm.tqdm(img_list):
        try:
            result,meta = inference_detector(model, img)
            result,meta = result[1],meta[0][0]
            results_all.append(copy.deepcopy(result))
            if result.shape[0]>0:
                result = result[result[:,-1]>args.score_thr]


            #### show the results
            out_file = os.path.join(args.out_file,os.path.split(img)[-1])
            # img_show = cv2.imread(img)
            # for rect in result:
            #     pt1 = np.array(rect[0:2],dtype=int)
            #     pt2 = np.array(rect[2:4],dtype=int)
            #     img_show=cv2.rectangle(img_show,pt1=pt1, pt2=pt2, color=(0,0,255),thickness=4)
            # cv2.imwrite(out_file,img_show)
        except:
            print(img + 'failed')
    predictions = []
    for res in results_all:
        posN = np.sum((res[:,-1]>=args.score_thr)*1)
        if posN:
            predictions.append(1)
        else:
            predictions.append(0)
    predictions = np.array(predictions,dtype=int)
    labels = [v[-10] for v in img_list]
    for i,v in enumerate(labels):
        if v=="+":
            labels[i] = 1
        elif v=="-":
            labels[i] = 0
        else:
            raise("error")
    labels = np.array(labels,dtype=int)
    precision = np.sum(labels*predictions)/np.sum(predictions)
    recall = np.sum(labels*predictions)/np.sum(labels)
    f1 = 2*precision*recall/(precision + recall)
    acc = np.sum((predictions==labels)*1.0)/labels.shape[0]
    print('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f'%(precision,recall,f1,acc))
    with open(args.out_file+".txt",'w+') as f:
        f.write('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f'%(precision,recall,f1,acc))

    pass




async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
