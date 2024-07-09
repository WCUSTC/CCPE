# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os
import os.path
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset

import matplotlib
import matplotlib.pyplot as plt
import cv2

@DATASETS.register_module()
class CocoGoggleDataset(CocoDataset):

    CLASSES = ('no_goggle', 'wear_goggle')

    PALETTE = [(220, 20, 60), (119, 11, 32)]

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,work_dir='./work_dir/'):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        print('---------------------------------BBox Metrics ---------------------------------')
        evaluator_without = DetectionRectsEval(iou_threshold=0.5)
        evaluator_with = DetectionRectsEval(iou_threshold=0.5)
        evaluator_class_agnostic = DetectionRectsEval(iou_threshold=0.5)
        score_threshold = 0.5
        pred_withoutN = []
        for i, pred in enumerate( results):
            ann = coco_gt.imgToAnns[i]
            ann_without = [v['bbox'] for v in ann if v['category_id']==1]
            ann_without = [np.array([v[0],v[1],v[2]+v[0],v[3]+v[1]]) for v in ann_without]
            pred_without = list(pred[0][pred[0][:, -1] > score_threshold][:, 0:-1])
            pred_withoutN.append(len(pred_without))
            evaluator_without.updata(gt_rects=ann_without, pred_rects=pred_without)

            ann_with = [v['bbox'] for v in ann if v['category_id']==2]
            ann_with = [np.array([v[0],v[1],v[2]+v[0],v[3]+v[1]]) for v in ann_with]

            pred_with = list(pred[1][pred[1][:, -1] > score_threshold][:, 0:-1])
            evaluator_with.updata(gt_rects=ann_with, pred_rects=pred_with)
            evaluator_class_agnostic.updata(gt_rects=ann_without+ann_with, pred_rects=pred_without+pred_with)
        evaluator_without.get()
        evaluator_with.get()
        evaluator_class_agnostic.get()
        ALL_tp = evaluator_without.tp+evaluator_with.tp
        ALL_fp = evaluator_without.fp+evaluator_with.fp
        ALL_fn = evaluator_without.fn+evaluator_with.fn
        ALL_recall = (ALL_tp)/(ALL_tp+ALL_fn + 1E-8)
        ALL_precision = (ALL_tp)/(ALL_tp+ALL_fp + 1E-8)
        ALL_f1 = ALL_recall*ALL_precision*2/(ALL_recall+ALL_precision + 1e-10)
        print('ALL: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
              % (ALL_tp, ALL_fp, ALL_fn, ALL_recall, ALL_precision, ALL_f1))
        print('no_goggle: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
              % (evaluator_without.tp, evaluator_without.fp, evaluator_without.fn, evaluator_without.recall,
                 evaluator_without.precision, evaluator_without.f1))
        print('wear_goggle: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
              % (evaluator_with.tp, evaluator_with.fp, evaluator_with.fn, evaluator_with.recall,
                 evaluator_with.precision, evaluator_with.f1))
        print('class_agnostic: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
              % (evaluator_class_agnostic.tp, evaluator_class_agnostic.fp, evaluator_class_agnostic.fn, evaluator_class_agnostic.recall,
                 evaluator_class_agnostic.precision, evaluator_class_agnostic.f1))

        get_conditional_confusion_matrix(evaluator_class_agnostic,evaluator_without,evaluator_with)

        save_path = os.path.join(work_dir, 'show')+'/'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        for i, pred in enumerate( results):
            pred_without = np.array(pred[0][pred[0][:, -1] > score_threshold][:, 0:-1])
            pred_with = np.array(pred[1][pred[1][:, -1] > score_threshold][:, 0:-1])
            img_path=self.img_prefix+coco_gt.imgs[i]['filename']
            img = cv2.imread(img_path)
            for box in pred_with:
                pt1 = np.array(box[0:2], dtype=int)
                pt2 = np.array(box[2:0:-1], dtype=int)
                pt3 = np.array(box[2:4], dtype=int)
                pt4 = np.array(box[0:4:3], dtype=int)
                img = cv2.rectangle(img, pt1, pt3, (0, 255, 0), 2)
            for box in pred_without:
                pt1 = np.array(box[0:2], dtype=int)
                pt2 = np.array(box[2:0:-1], dtype=int)
                pt3 = np.array(box[2:4], dtype=int)
                pt4 = np.array(box[0:4:3], dtype=int)
                img = cv2.rectangle(img, pt1, pt3, (0, 0, 255), 2)
            cv2.imwrite(save_path+img_path.split('/')[-1],img)



        # save_path = os.path.join(work_dir, 'errors')+'/'
        # if os.path.exists(save_path) is False:
        #     os.makedirs(save_path)
        # for i, pred in enumerate( results):
        #     ann = coco_gt.imgToAnns[i]
        #     ann_without = [v['bbox'] for v in ann if v['category_id']==1]
        #     ann_without = np.array([np.array([v[0],v[1],v[2]+v[0],v[3]+v[1]]) for v in ann_without])
        #     pred_without = np.array(pred[0][pred[0][:, -1] > score_threshold][:, 0:-1])
        #     img_path=self.img_prefix+coco_gt.imgs[i]['filename']
        #
        #     plt.figure('%06d' % i)
        #     img_show = matplotlib.image.imread(img_path)
        #     plt.imshow(img_show,)
        #     plt.axis('off')
        #     pred_without_tp = pred_without[evaluator_without.tp_indexes[i]]
        #     pred_without_fp = pred_without[evaluator_without.fp_indexes[i]]
        #     pred_without_fn = ann_without[evaluator_without.fn_indexes[i]]
        #     show_bboxes(pred_without_tp,color_line='r-')
        #     show_bboxes( pred_without_fp, color_line='r--')
        #     show_bboxes( pred_without_fn, color_line='r:')
        #
        #     ann_with = [v['bbox'] for v in ann if v['category_id']==2]
        #     ann_with = np.array([np.array([v[0],v[1],v[2]+v[0],v[3]+v[1]]) for v in ann_with])
        #     pred_with = np.array(pred[1][pred[1][:, -1] > score_threshold][:, 0:-1])
        #     pred_with_tp = pred_with[evaluator_with.tp_indexes[i]]
        #     pred_with_fp = pred_with[evaluator_with.fp_indexes[i]]
        #     pred_with_fn = ann_with[evaluator_with.fn_indexes[i]]
        #     show_bboxes(pred_with_tp,color_line='g-')
        #     show_bboxes( pred_with_fp, color_line='g--')
        #     show_bboxes( pred_with_fn, color_line='g:')
        #     plt.savefig(save_path+coco_gt.imgs[i]['filename'].split('/')[-1], dpi=200)
        #     plt.close()

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

def show_bboxes(bboxes,color_line):
    for box in bboxes:
        pt1 = np.array(box[0:2],dtype=int)
        pt2 = np.array(box[2:0:-1],dtype=int)
        pt3 = np.array(box[2:4],dtype=int)
        pt4 = np.array(box[0:4:3],dtype=int)
        plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],color_line)
        plt.plot([pt2[0], pt3[0]], [pt2[1], pt3[1]], color_line)
        plt.plot([pt3[0], pt4[0]], [pt3[1], pt4[1]], color_line)
        plt.plot([pt4[0], pt1[0]], [pt4[1], pt1[1]], color_line)


def get_conditional_confusion_matrix(evaluator_class_agnostic,evaluator_without,evaluator_with):
    no_pred_with= 0
    with_pred_no = 0

    no_pred_no = 0
    with_pred_with = 0
    for i,tp_agnostic in enumerate(evaluator_class_agnostic.tp_indexes):
        if tp_agnostic.shape[0]==0:
            continue
        noN_pred = evaluator_without.tp_indexes[i].shape[0] + evaluator_without.fp_indexes[i].shape[0]
        withN_pred = evaluator_with.tp_indexes[i].shape[0] + evaluator_with.fp_indexes[i].shape[0]
        noN_gt = evaluator_without.tp_indexes_gt[i].shape[0] + evaluator_without.fn_indexes[i].shape[0]
        withN_gt = evaluator_with.tp_indexes_gt[i].shape[0] + evaluator_with.fn_indexes[i].shape[0]
        '''
        不能用排除法，因为预测的no与with可能是相同的框，evaluator_class_agnostic会只选择一个框匹配。
        造成evaluator_class_agnostic中来源于某类的tp还没有此类的tp数量多。
        '''
        # ##来源于no的无类别tp，减去no有类别tp，就是预测为no的实际为with的
        # with_pred_no +=( np.count_nonzero(tp_agnostic<noN) - evaluator_without.tp_indexes[i].shape[0])
        # ##来源于with的无类别tp，减去with有类别tp，就是预测为with的实际为no的
        # no_pred_with += (np.count_nonzero(tp_agnostic>=noN) - evaluator_with.tp_indexes[i].shape[0])
        '''正面计算'''
        label_pred = tp_agnostic>=noN_pred
        label_gt =  evaluator_class_agnostic.tp_indexes_gt[i] >= noN_gt
        with_pred_no += np.count_nonzero((label_pred==False) * (label_gt==True))
        no_pred_with += np.count_nonzero((label_pred==True) * (label_gt==False))

        no_pred_no +=np.count_nonzero((label_pred==False) * (label_gt==False))
        with_pred_with += np.count_nonzero((label_pred==True) * (label_gt==True))
    print('no_pred_no = %04d , no_pred_wi = %04d '%(no_pred_no,no_pred_with))
    print('wi_pred_no = %04d , wi_pred_wi = %04d '%(with_pred_no,with_pred_with))



import scipy
class DetectionRectsEval:
    def __init__(self,iou_threshold=0.1):
        self.tp=0
        self.fp=0
        self.fn=0
        self.iou_threshold = iou_threshold
        self.tp_indexes = []
        self.fp_indexes = []
        self.fn_indexes = []
        self.tp_indexes_gt = []


    def cal_iou_mat (self,rects1,rects2):
        iou_mat =np.zeros((len(rects1),len(rects2)),dtype=float)
        bboxes1 = np.array(rects1)
        bboxes2 = np.array(rects2)
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] ) * ( bboxes1[:, 3] - bboxes1[:, 1] )
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] ) * (  bboxes2[:, 3] - bboxes2[:, 1] )
        for i in range(bboxes1.shape[0]):
            x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
            y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
            x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
            y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
            overlap = np.maximum(x_end - x_start , 0) * np.maximum( y_end - y_start , 0)
            union = area1[i] + area2 - overlap
            union = np.maximum(union, 1e-10)
            iou_mat[i, :] = overlap / union
        return iou_mat


    def updata(self,gt_rects,pred_rects):
        '''rects的格式为[x_min,y_min,x_max,y_max]'''
        if len(gt_rects)+len(pred_rects)==0:
            self.gt_matched = np.array([],dtype=bool)
            self.pred_matched = np.array([],dtype=bool)
            self.tp_indexes.append(np.zeros([0],dtype=int))
            self.fp_indexes.append(np.zeros([0],dtype=int))
            self.fn_indexes.append(np.zeros([0],dtype=int))
            self.tp_indexes_gt.append(np.zeros([0],dtype=int))
            return
        if len(gt_rects)==0:
            self.fp += len(pred_rects)
            self.gt_matched = np.array([],dtype=bool)
            self.pred_matched = np.array([False]*len(pred_rects),dtype=bool)
            self.tp_indexes.append(np.zeros([0],dtype=int))
            self.fp_indexes.append(np.array(range(len(pred_rects))))
            self.fn_indexes.append(np.zeros([0],dtype=int))
            self.tp_indexes_gt.append(np.zeros([0],dtype=int))
            return
        if len(pred_rects)==0:
            self.fn +=len(gt_rects)
            self.gt_matched = np.array([False]*len(gt_rects),dtype=bool)
            self.pred_matched = np.array([],dtype=bool)
            self.tp_indexes.append(np.zeros([0],dtype=int))
            self.fp_indexes.append(np.zeros([0],dtype=int))
            self.fn_indexes.append(np.array(range(len(gt_rects))))
            self.tp_indexes_gt.append(np.zeros([0],dtype=int))
            return
        iou_mat = self.cal_iou_mat(gt_rects,pred_rects)
        match_index_list = scipy.optimize.linear_sum_assignment(cost_matrix=1-iou_mat)
        matched_mat = np.zeros_like(iou_mat)
        matched_mat[match_index_list[0],match_index_list[1]] =1
        iou_mat_matched = iou_mat * matched_mat
        iou_mat_matched_T = iou_mat_matched>=self.iou_threshold
        self.gt_matched = np.max(iou_mat_matched_T,axis=1)
        self.pred_matched = np.max(iou_mat_matched_T,axis=0)
        self.tp +=np.sum(self.pred_matched*1)
        self.fp +=(len(pred_rects) - np.sum(self.pred_matched*1))
        self.fn +=(len(gt_rects) - np.sum(self.pred_matched*1))

        self.tp_indexes.append(np.argwhere(self.pred_matched)[:,0])
        self.fp_indexes.append(np.argwhere(~self.pred_matched)[:,0])
        self.fn_indexes.append(np.argwhere(~self.gt_matched)[:,0])
        self.tp_indexes_gt.append(np.argwhere(self.gt_matched)[:,0])

    def get(self):
        self.recall = self.tp/(1e-10+self.tp+self.fn)
        self.precision = self.tp/(1e-10+self.tp+self.fp)
        self.f1 = self.recall*self.precision*2/(self.recall+self.precision + 1e-10)