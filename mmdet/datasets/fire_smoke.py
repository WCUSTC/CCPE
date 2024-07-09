# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from collections import OrderedDict
import torch
import tqdm
from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
import os.path as osp
import mmcv,cv2,scipy
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from mmcv.parallel import DataContainer as DC

points =list(np.linspace(0,0.05,500,endpoint=False)) + list(np.linspace(0.05,0.95,500,endpoint=False)) + list(np.linspace(0.95,1,500,endpoint=False))
class DetectionRectsEval:
    def __init__(self,iou_threshod=0.1):
        self.tp=0
        self.fp=0
        self.fn=0
        self.iou_threshod = iou_threshod
        self.pred_matched_all = []

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
            return
        if len(gt_rects)==0:
            self.fp += len(pred_rects)
            self.gt_matched = np.array([],dtype=bool)
            self.pred_matched = np.array([False]*len(pred_rects),dtype=bool)
            return
        if len(pred_rects)==0:
            self.fn +=len(gt_rects)
            self.gt_matched = np.array([False]*len(gt_rects),dtype=bool)
            self.pred_matched = np.array([],dtype=bool)
            return
        iou_mat = self.cal_iou_mat(gt_rects,pred_rects)
        match_index_list = scipy.optimize.linear_sum_assignment(cost_matrix=1-iou_mat)
        matched_mat = np.zeros_like(iou_mat)
        matched_mat[match_index_list[0],match_index_list[1]] =1
        iou_mat_matched = iou_mat * matched_mat
        iou_mat_matched_T = iou_mat_matched>=self.iou_threshod
        self.gt_matched = np.max(iou_mat_matched_T,axis=1)
        self.pred_matched = np.max(iou_mat_matched_T,axis=0)
        self.pred_matched_all.append(copy.deepcopy(self.pred_matched))
        self.tp +=np.sum(self.pred_matched*1)
        self.fp +=len(pred_rects) - np.sum(self.pred_matched*1)
        self.fn +=len(gt_rects) - np.sum(self.pred_matched*1)


    def get(self):
        self.recall = self.tp/(1e-10+self.tp+self.fn)
        self.precision = self.tp/(1e-10+self.tp+self.fp)
        self.f1 = self.recall*self.precision*2/(self.recall+self.precision + 1e-10)


@DATASETS.register_module()
class FireSmokeDataset(XMLDataset):
    CLASSES = ('fire', 'smoke')
    PALETTE = [(0, 255, 255), (255, 0, 0)]
    def __init__(self, **kwargs):
        super(FireSmokeDataset, self).__init__(**kwargs)
        self.video_dict = self.obtain_video_dict_accordname()
        self.data_statistic()
        self.year = 2007

    def data_statistic(self):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        label_fire = np.array([np.sum(ann['labels']==0) for ann in annotations],dtype=int)
        label_fire = np.array((label_fire>=1)*1,dtype=int)
        label_smoke = np.array([np.sum(ann['labels']==1) for ann in annotations],dtype=int)
        label_smoke = np.array((label_smoke>=1)*1,dtype=int)

        gt_video_fire, gt_video_smoke = [], []
        for video in self.video_dict.keys():
            ids = np.array(self.video_dict[video])
            gt_fire = np.max(label_fire[ids])
            gt_smoke = np.max(label_smoke[ids])
            gt_video_fire.append(gt_fire)
            gt_video_smoke.append(gt_smoke)
        gt_video_fire, gt_video_smoke = np.array(gt_video_fire, dtype=int), np.array(gt_video_smoke, dtype=int)

        print(
            "****   Total Images: %08d     ImagesWithFire: %08d     ImagesWithSmoke: %08d     ImagesWithFireAndSmoke: %08d     ImagesNeg: %08d  ********" % (
                len(annotations), np.sum(label_fire), np.sum(label_smoke),
                np.sum(((label_fire + label_smoke) >= 2) * 1),
                len(annotations) - np.sum(((label_fire + label_smoke) >= 1) * 1)
            ))

        print(
            "****   Total Videos: %08d     VideosWithFire: %08d     VideosWithSmoke: %08d     VideosWithFireAndSmoke: %08d     VideosNeg: %08d  ********" % (
                len(self.video_dict), np.sum(gt_video_fire), np.sum(gt_video_smoke),
                np.sum(((gt_video_fire + gt_video_smoke) >= 2) * 1),
                len(self.video_dict) - np.sum(((gt_video_fire + gt_video_smoke) >= 1) * 1)
            ))

    def obtain_video_dict_accordname(self):
        video_dict={}
        for i,name in enumerate(self.data_infos):
            name = '--'.join(name['id'].split('--')[:-1])
            if name not in video_dict.keys():
                video_dict[name] = [i]
            else:
                video_dict[name].append(i)
        return video_dict


    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        print('Begin to load %s-----'%ann_file)
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        # img_ids = [v for v in img_ids if v+'.jpg' in os.listdir(r'D:\学术管理\王冲\ContrastSwin\Vis_comp\错例图片')]

        for img_id in tqdm.tqdm(img_ids):
            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
            if osp.exists(xml_path) is False:
                img_path = osp.join(self.img_prefix,self.img_subdir,f'{img_id}.jpg')
                if osp.exists(img_path) is False:
                    continue
                img = cv2.imread(img_path)
                height,width,C =img.shape
            else:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                if size is not None:
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                else:
                    img_path = osp.join(self.img_prefix, filename)
                    img = Image.open(img_path)
                    width, height = img.size
            data_infos.append( dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos


    def _filter_imgs(self, min_size=32):
        """Filter images too small (or without annotation)."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        if osp.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.CLASSES:
                    continue
                label = self.cat2label[name]
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
                bnd_box = obj.find('bndbox')
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.1,
                 score_thr=0.1,
                 scale_ranges=None):
        if  isinstance(metric, str):
            metric_list = [metric]
        else:
            metric_list = copy.deepcopy(metric)
        allowed_metrics = ['mAP', 'bbox','fireACC_img','smokeACC_img']
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        label_fire = np.array([np.sum(ann['labels']==0) for ann in annotations],dtype=int)
        label_fire = np.array((label_fire>=1)*1,dtype=int)
        label_smoke = np.array([np.sum(ann['labels']==1) for ann in annotations],dtype=int)
        label_smoke = np.array((label_smoke>=1)*1,dtype=int)

        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        eval_results = OrderedDict()
        eval_results['score_thr'] = score_thr

        for metric in metric_list:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
            if metric == 'mAP':
                assert isinstance(iou_thrs, list)
                if self.year == 2007:
                    ds_name = 'voc07'
                else:
                    ds_name = self.CLASSES
                mean_aps = []
                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, ap_res = eval_map(
                        results,
                        annotations,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset=ds_name,
                        logger=logger,
                        use_legacy_coordinate=True)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
                eval_results.move_to_end('mAP', last=False)
                eval_results['AP'] = ap_res

            elif metric=='bbox':
                eval_results['bbox'] = {}
                for score_thr in points:
                    current_result={}
                    evaluator_fire = DetectionRectsEval(iou_threshod=0.1)
                    evaluator_smoke = DetectionRectsEval(iou_threshod=0.1)
                    for ann, pred in zip(annotations,results):
                        ann_fire = list(ann['bboxes'][ann['labels']==0])
                        pred_fire = list(pred[0][pred[0][:,-1]>score_thr][:,0:-1])
                        evaluator_fire.updata(gt_rects=ann_fire, pred_rects=pred_fire)

                        ann_smoke = list(ann['bboxes'][ann['labels']==1])
                        pred_smoke = list(pred[1][pred[1][:,-1]>score_thr])
                        evaluator_smoke.updata(gt_rects=ann_smoke, pred_rects=pred_smoke)

                    evaluator_fire.get()
                    evaluator_smoke.get()
                    print('score_thr:%.4f  fire: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
                          % (score_thr,evaluator_fire.tp, evaluator_fire.fp, evaluator_fire.fn, evaluator_fire.recall,
                             evaluator_fire.precision, evaluator_fire.f1))
                    print('score_thr:%.4f   smoke: tp=%05d,  fp=%05d,   fn=%05d,   recall=%.4f,  precision=%.4f,   f1=%.4f'
                          % (score_thr,evaluator_smoke.tp, evaluator_smoke.fp, evaluator_smoke.fn, evaluator_smoke.recall,
                             evaluator_smoke.precision, evaluator_smoke.f1))

                    current_result['fire_TP'] = evaluator_fire.tp
                    current_result['fire_FP'] = evaluator_fire.fp
                    current_result['fire_FN'] = evaluator_fire.fn
                    current_result['fire_recall'] = evaluator_fire.recall,
                    current_result['fire_precision'] = evaluator_fire.precision
                    current_result['fire_f1'] = evaluator_fire.f1
                    current_result['smoke_TP'] = evaluator_smoke.tp
                    current_result['smoke_FP'] = evaluator_smoke.fp
                    current_result['smoke_FN'] = evaluator_smoke.fn
                    current_result['smoke_recall'] = evaluator_smoke.recall,
                    current_result['smoke_precision'] = evaluator_smoke.precision
                    current_result['smoke_f1'] = evaluator_smoke.f1
                    eval_results['bbox']['%f' % score_thr] = current_result
            elif metric=='fireACC_img':
                gt_img = label_fire
                eval_results['img_fire'] = {}
                for score_thr in points:
                    current_result={}
                    pred_img = []
                    score_img = []
                    for img_i,pred in enumerate(results):
                        pred = pred[0]
                        if pred.shape[0]==0:
                            pred_img.append(0)
                            score_img.append(0.0)
                            continue
                        else:
                            score_img.append(np.max(pred[:,-1]))
                        whithout_pos = np.alltrue(pred[:,-1]<score_thr)
                        if whithout_pos:
                            pred_img.append(0)
                        else:
                            pred_img.append(1)
                    pred_img = np.array(pred_img,dtype=int)
                    score_img = np.array(score_img)
                    eval_results['img_fire_score']=copy.deepcopy(score_img)
                    eval_results['img_fire_gt']=copy.deepcopy(gt_img)
                    current_result['img_fire_TP'] = np.sum(pred_img*gt_img )
                    current_result['img_fire_FP'] = np.sum(pred_img*(1-gt_img) )
                    current_result['img_fire_FN'] = np.sum((1-pred_img)*gt_img )
                    current_result['img_fire_TN'] = np.sum((1-pred_img)*(1-gt_img) )
                    current_result['img_fire_ACC'] = np.sum(pred_img==gt_img )/gt_img.shape[0]
                    current_result['img_fire_precision'] = current_result['img_fire_TP']/(current_result['img_fire_TP']+current_result['img_fire_FP'] +1e-10)
                    current_result['img_fire_recall'] = current_result['img_fire_TP']/(current_result['img_fire_TP']+current_result['img_fire_FN'] +1e-10)
                    current_result['img_fire_f1'] = current_result['img_fire_recall'] * current_result['img_fire_precision']*2/(current_result['img_fire_recall'] + current_result['img_fire_precision'] + 1e-10)

                    pred_video, gt_video,score_video = [], [], []
                    for video in self.video_dict.keys():
                        ids = np.array(self.video_dict[video])
                        pred = np.max(pred_img[ids])
                        gt = np.max(gt_img[ids])
                        pred_video.append(pred)
                        gt_video.append(gt)
                        score= np.max(score_img[ids])
                        score_video.append(score)
                    pred_video, gt_video = np.array(pred_video,dtype=int),np.array(gt_video,dtype=int)

                    eval_results['vider_fire_score'] = np.array(score_video)
                    eval_results['vider_fire_gt'] = np.array(gt_video)
                    current_result['video_fire_TP'] = np.sum(pred_video * gt_video)
                    current_result['video_fire_FP'] = np.sum(pred_video * (1 - gt_video))
                    current_result['video_fire_FN'] = np.sum((1 - pred_video) * gt_video)
                    current_result['video_fire_TN'] = np.sum((1 - pred_video) * (1 - gt_video))
                    current_result['video_fire_ACC'] = np.sum(pred_video == gt_video) / gt_video.shape[0]
                    current_result['video_fire_precision'] = current_result['video_fire_TP'] / ( current_result['video_fire_TP'] + current_result['video_fire_FP'] + 1e-10)
                    current_result['video_fire_recall'] = current_result['video_fire_TP'] / ( current_result['video_fire_TP'] + current_result['video_fire_FN'] + 1e-10)
                    current_result['video_fire_f1'] = current_result['video_fire_recall'] * current_result['video_fire_precision'] * 2 / \
                                                       (current_result['video_fire_recall'] + current_result[ 'video_fire_precision'] + 1e-10)
                    eval_results['img_fire']['%f'%score_thr] = current_result
            elif metric=='smokeACC_img':
                gt_img = label_smoke
                eval_results['img_smoke'] = {}
                for score_thr in points:
                    pred_img = []
                    score_img = []
                    current_result={}
                    for img_i, pred in enumerate(results):
                        pred = pred[1]
                        if pred.shape[0] == 0:
                            pred_img.append(0)
                            score_img.append(0.0)
                            continue
                        else:
                            score_img.append(np.max(pred[:, -1]))
                        whithout_pos = np.alltrue(pred[:, -1] < score_thr)
                        if whithout_pos:
                            pred_img.append(0)
                        else:
                            pred_img.append(1)
                    pred_img = np.array(pred_img, dtype=int)
                    score_img = np.array(score_img,dtype=float)
                    eval_results['img_smoke_score'] = score_img
                    eval_results['img_smoke_gt'] = gt_img
                    current_result['img_smoke_TP'] = np.sum(pred_img * gt_img)
                    current_result['img_smoke_FP'] = np.sum(pred_img * (1 - gt_img))
                    current_result['img_smoke_FN'] = np.sum((1 - pred_img) * gt_img)
                    current_result['img_smoke_TN'] = np.sum((1 - pred_img) *  (1 - gt_img))
                    current_result['img_smoke_ACC'] = np.sum(pred_img == gt_img) / gt_img.shape[0]
                    current_result['img_smoke_precision'] = current_result['img_smoke_TP']/(current_result['img_smoke_TP']+current_result['img_smoke_FP'] +1e-10)
                    current_result['img_smoke_recall'] = current_result['img_smoke_TP']/(current_result['img_smoke_TP']+current_result['img_smoke_FN'] +1e-10)
                    current_result['img_smoke_f1'] = current_result['img_smoke_recall'] * current_result['img_smoke_precision']*2/(current_result['img_smoke_recall'] + current_result['img_smoke_precision'] + 1e-10)

                    pred_video,gt_video,score_video=[],[],[]
                    for video in self.video_dict.keys():
                        ids = np.array(self.video_dict[video])
                        pred = np.max(pred_img[ids])
                        gt = np.max(gt_img[ids])
                        pred_video.append(pred)
                        gt_video.append(gt)
                        score_video.append(np.max(score_img[ids]))

                    pred_video, gt_video = np.array(pred_video,dtype=int),np.array(gt_video,dtype=int)
                    score_video = np.array(score_video)

                    eval_results['video_smoke_score'] = score_video
                    eval_results['video_smoke_gt'] = gt_video
                    current_result['video_smoke_TP'] = np.sum(pred_video * gt_video)
                    current_result['video_smoke_FP'] = np.sum(pred_video * (1 - gt_video))
                    current_result['video_smoke_FN'] = np.sum((1 - pred_video) * gt_video)
                    current_result['video_smoke_TN'] = np.sum((1 - pred_video) * (1 - gt_video))
                    current_result['video_smoke_ACC'] = np.sum(pred_video == gt_video) / gt_video.shape[0]
                    current_result['video_smoke_precision'] = current_result['video_smoke_TP'] / ( current_result['video_smoke_TP'] + current_result['video_smoke_FP'] + 1e-10)
                    current_result['video_smoke_recall'] = current_result['video_smoke_TP'] / ( current_result['video_smoke_TP'] + current_result['video_smoke_FN'] + 1e-10)
                    current_result['video_smoke_f1'] = current_result['video_smoke_recall'] * current_result[ 'video_smoke_precision'] * 2 /  \
                                                       (current_result['video_smoke_recall'] + current_result['video_smoke_precision'] + 1e-10)
                    eval_results['img_smoke']['%f'%score_thr] = current_result
        return eval_results


@DATASETS.register_module()
class FireSmokeDatasetFIgLibMultiFrames(FireSmokeDataset):
    def __init__(self, minutes, **kwargs):
        super(FireSmokeDatasetFIgLibMultiFrames, self).__init__(**kwargs)
        self.context_frames = self.get_previous_index(self.data_infos, minutes=minutes)
        pass
    def get_previous_index(self, data_infos,  minutes:list):
        '''
        Args:
            data_infos:
            minutes: 是一个列表，每个当前帧可能用多个上下文帧。列表内，负值代表过去，正值代表未来
        Returns:
        '''
        '''先产生每个视频有哪些帧（为视频内排序做准备），每个帧在哪个序号'''
        video_dict = {}
        file2index ={}
        for index in range(len(data_infos)):
            filename = os.path.basename(data_infos[index]["filename"])
            file2index.update({filename:index})
            videoname,framename = filename.split("---")
            if videoname not in video_dict.keys():
                video_dict.update({videoname:[filename]})
            else:
                video_dict[videoname].append(filename)
        '''视频内对帧进行排序'''
        for video in video_dict.keys():
            video_dict[video].sort()
        self.video_dict = video_dict
        '''每个帧输出对应的历史帧序号。并且在self.data_infos[index]中加filenamelist键值对'''
        self.context_frames= []
        for index in range(len(data_infos)):
            filename = os.path.basename(data_infos[index]["filename"])
            videoname, framename = filename.split("---")
            frames = []
            filenamelist = []
            id_in_video = video_dict[videoname].index(filename)
            for minute in minutes:
                id_ = id_in_video + minute
                if id_>=0 and id_<len(video_dict[videoname]):
                    id_in_data_infos = file2index[video_dict[videoname][id_]]
                    frames.append(id_in_data_infos)
                    filenamelist.append(self.data_infos[id_in_data_infos]["filename"])
                else:
                    frames.append(None)
                    filenamelist.append(None)
            self.context_frames.append(copy.deepcopy(frames))
            self.data_infos[index]['filenamelist'] = filenamelist
        return self.context_frames

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_test_img(idx)
            img_concat = data["img"][0]
            for i in range(100):
                if "img_%03d"%i in data.keys():
                    img_concat = torch.cat([img_concat,data["img_%03d"%i][0]])
                    data.pop("img_%03d"%i)
                else:
                    break
            data["img"] = [img_concat.float()]
            return data
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            img_concat = data["img"].data
            for i in range(100):
                if "img_%03d"%i in data.keys():
                    img_concat = torch.cat([img_concat,data["img_%03d"%i]])
                    data.pop("img_%03d"%i)
                else:
                    break
            data["img"] = DC(img_concat, padding_value=0, stack=True)
            return data


    def evaluate(self,
                 results,
                 metric='xxx',
                 logger=None,
                 score_thr=0.1,
                 out_file = "",
                 scale_ranges=None,remove1video=False):
        img_list = [os.path.split(v["filename"])[-1] for v in self.data_infos]
        results_all = results
        predictions = []
        for res in results_all:
            res = res[1]
            posN = np.sum((res[:, -1] >= score_thr) * 1)
            if posN:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions, dtype=int)
        labels = [v[-10] for v in img_list]
        for i, v in enumerate(labels):
            if v == "+":
                labels[i] = 1
            elif v == "-":
                labels[i] = 0
            else:
                raise ("error")
        labels = np.array(labels, dtype=int)
        if remove1video:
            eval_results_all = []
            for video_removed in self.video_dict.keys():
                video_dict = copy.deepcopy(self.video_dict)
                video_dict.pop(video_removed)
                indexes_removed = [img_list.index(name) for name in self.video_dict[video_removed]]
                eval_results = self.get_metrics(labels=np.array([v for i, v in enumerate(labels) if (i not in indexes_removed)]),
                                                predictions=np.array([v for i, v in enumerate(predictions) if (i not in indexes_removed)]),
                                                img_list=[v for i, v in enumerate(img_list) if (i not in indexes_removed)],
                                                video_dict =video_dict )
                eval_results_all.append(eval_results)
            acc_all = np.array([v["acc"] for v in eval_results_all])
            acc_min_id = np.argmin(acc_all,axis=0)
            eval_results = eval_results_all[acc_min_id]
        else:
            eval_results=self.get_metrics(labels,predictions,img_list,self.video_dict)
        return eval_results


    def get_metrics(self,labels,predictions,img_list,video_dict):
        precision = np.sum(labels * predictions) / np.sum(predictions)
        recall = np.sum(labels * predictions) / np.sum(labels)
        f1 = 2 * precision * recall / (precision + recall)
        acc = np.sum((predictions == labels) * 1.0) / labels.shape[0]
        TTD = []
        for key in video_dict.keys():
            label_single_video = [labels[img_list.index(name)] for name in video_dict[key]]
            pred_single_video = [predictions[img_list.index(name)] for name in video_dict[key]]
            fist_tp = np.array(label_single_video) * np.array(pred_single_video)
            fist_tp = np.where(fist_tp)[0][0]
            zero_id = 0
            for i,name in enumerate(video_dict[key]):
                order = name.split("_")[-1]
                if order[0]=="+":
                    zero_id = i
                    break
            TTD.append(fist_tp-zero_id)
        TTD_ave = np.mean(np.array(TTD))
        print('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f  TTD:%.4f' % (precision, recall, f1, acc,TTD_ave))
        # with open(out_file, 'a+') as f:
        #     f.write('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f  TTD:%.4f \n' % (precision, recall, f1, acc, TTD_ave))
        eval_results={"precision":precision,"recall":recall,"f1":f1,"acc":acc,"TTD" : TTD_ave}
        return eval_results



@DATASETS.register_module()
class FireSmokeDatasetFIgLib(FireSmokeDataset):
    def __init__(self, **kwargs):
        super(FireSmokeDatasetFIgLib, self).__init__(**kwargs)
        self.context_frames = self.get_previous_index(self.data_infos, minutes=[-1])
        pass

    def get_previous_index(self, data_infos,  minutes:list):
        '''
        Args:
            data_infos:
            minutes: 是一个列表，每个当前帧可能用多个上下文帧。列表内，负值代表过去，正值代表未来
        Returns:
        '''
        '''先产生每个视频有哪些帧（为视频内排序做准备），每个帧在哪个序号'''
        video_dict = {}
        file2index ={}
        for index in range(len(data_infos)):
            filename = os.path.basename(data_infos[index]["filename"])
            file2index.update({filename:index})
            videoname,framename = filename.split("---")
            if videoname not in video_dict.keys():
                video_dict.update({videoname:[filename]})
            else:
                video_dict[videoname].append(filename)
        '''视频内对帧进行排序'''
        for video in video_dict.keys():
            video_dict[video].sort()
        self.video_dict = video_dict
        '''每个帧输出对应的历史帧序号。并且在self.data_infos[index]中加filenamelist键值对'''
        self.context_frames= []
        for index in range(len(data_infos)):
            filename = os.path.basename(data_infos[index]["filename"])
            videoname, framename = filename.split("---")
            frames = []
            filenamelist = []
            id_in_video = video_dict[videoname].index(filename)
            for minute in minutes:
                id_ = id_in_video + minute
                if id_>=0 and id_<len(video_dict[videoname]):
                    id_in_data_infos = file2index[video_dict[videoname][id_]]
                    frames.append(id_in_data_infos)
                    filenamelist.append(self.data_infos[id_in_data_infos]["filename"])
                else:
                    frames.append(None)
                    filenamelist.append(None)
            self.context_frames.append(copy.deepcopy(frames))
            self.data_infos[index]['filenamelist'] = filenamelist
        return self.context_frames


    def evaluate(self,
                 results,
                 metric='xxx',
                 logger=None,
                 score_thr=0.1,
                 out_file = "",
                 scale_ranges=None):
        img_list = [os.path.split(v["filename"])[-1] for v in self.data_infos]
        results_all = results
        predictions = []
        for res in results_all:
            res = res[1]
            posN = np.sum((res[:, -1] >= score_thr) * 1)
            if posN:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions, dtype=int)
        labels = [v[-10] for v in img_list]
        for i, v in enumerate(labels):
            if v == "+":
                labels[i] = 1
            elif v == "-":
                labels[i] = 0
            else:
                raise ("error")
        labels = np.array(labels, dtype=int)
        precision = np.sum(labels * predictions) / np.sum(predictions)
        recall = np.sum(labels * predictions) / np.sum(labels)
        f1 = 2 * precision * recall / (precision + recall)
        acc = np.sum((predictions == labels) * 1.0) / labels.shape[0]
        TTD = []
        for key in self.video_dict.keys():
            label_single_video = [labels[img_list.index(name)] for name in self.video_dict[key]]
            pred_single_video = [predictions[img_list.index(name)] for name in self.video_dict[key]]
            fist_tp = np.array(label_single_video) * np.array(pred_single_video)
            fist_tp = np.where(fist_tp)[0][0]
            zero_id = 0
            for i,name in enumerate(self.video_dict[key]):
                order = name.split("_")[-1]
                if order[0]=="+":
                    zero_id = i
                    break
            TTD.append(fist_tp-zero_id)
        TTD_ave = np.mean(np.array(TTD))


        print('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f  TTD:%.4f' % (precision, recall, f1, acc,TTD_ave))
        with open(out_file, 'a+') as f:
            f.write('precision:%.4f   recall:%.4f  f1:%.4f  acc:%.4f  TTD:%.4f \n' % (precision, recall, f1, acc, TTD_ave))
        eval_results={"precision":precision,"recall":recall,"f1":f1,"acc":acc,"TTD" : TTD_ave}
        return eval_results