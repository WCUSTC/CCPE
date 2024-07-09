# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import warnings
import cv2.gapi.wip.draw
import torch
import numpy as np
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import os, cv2,random
from mmdet.core.visualization import imshow_det_bboxes


class ClassificationModle(torch.nn.Module):
    def __init__(self,in_channels,out_channels=256,classN=2):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=out_channels,out_features=out_channels,bias=True),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=out_channels, out_features=classN, bias=True)
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def forward(self,feat):
        feat = self.stem(feat)
        feat = torch.nn.functional.adaptive_max_pool2d(feat,(1,1))
        feat = feat[...,0,0]
        logits = self.fc(feat)
        return logits


    def loss(self,logits, gt_labels):
        lable = torch.zeros(len(gt_labels),dtype=torch.long,device=logits.device)
        for i,v in enumerate(gt_labels):
            if v.shape[0]>0:
                lable[i] = 1
        loss = self.cross_entropy(logits,lable)
        return {'cls_loss':loss}

class SegmentationModle(torch.nn.Module):
    def __init__(self,in_channels,out_channels=256,classN=2):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1,bias=True),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=classN, kernel_size=(1, 1), stride=1, padding=0, bias=True),
        )
        self.cross_loss = torch.nn.CrossEntropyLoss()

    def forward(self,feat):
        feat = self.stem(feat)
        return feat

    def loss(self,logits,target):
        target = torch.nn.functional.upsample_bilinear(target,size=logits.size()[2:4])
        target = target[:,0,...].long()
        # loss = torch.nn.functional.cross_entropy(logits,target=target,weight=torch.from_numpy(np.array([0.1,1],dtype=np.float32)).to(logits.device),reduce=True)
        # logits = logits.contiguous().view(-1)
        # target = target.contiguous().view(-1)
        loss = self.cross_loss(logits,target=target)
        return {'seg_loss':loss}


    def loss1(self,predictive,target,ep = 1e-8):
        target = torch.nn.functional.upsample_bilinear(target,size=predictive.size()[2:4])
        intersection = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return {'seg_loss':loss}

@DETECTORS.register_module()
class ImageClassification(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 classN=2,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ImageClassification, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.classification_head = ClassificationModle(in_channels=neck.out_channels,out_channels=256,classN=classN)
        self.segmentation_head = SegmentationModle(in_channels=neck.out_channels,out_channels=256,classN=classN)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,gt_semantic_seg):

        super(ImageClassification, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        logits = self.classification_head(x[1])
        logits_seg = self.segmentation_head(x[0])
        losses = {}
        loss_cls = self.classification_head.loss(logits, gt_labels)
        foreground_score = torch.nn.functional.softmax(logits_seg,dim=1)[:,1:2,...]
        # loss_seg = self.dice_loss(foreground_score, gt_semantic_seg)
        loss_seg = self.segmentation_head.loss(logits_seg, gt_semantic_seg)
        losses.update(loss_cls)
        losses.update(loss_seg)

        if random.random()>=0.98:
            print('-----------------show-input-------------------------')
            self.show_input(img,  img_metas, gt_bboxes, gt_labels,foreground_score,gt_semantic_seg)
        return losses





    def aug_test(self, imgs, img_metas, rescale=False):
        """
        这个必须重写，因为父类有@abstractmethod 修饰
        """
        pass

    def show_input(self, img,  img_metas, gt_bboxes,gt_labels, foreground_score,gt_semantic_seg):
        if os.path.exists('../show_input') is False:
            os.makedirs('../show_input')
        img_show = img.permute(0,2,3,1)
        img_show = img_show.data.cpu().numpy()
        img_show = (img_show-img_show.min())*255/(img_show.max()-img_show.min())
        # img_show = (img_show-img_show.min())*255/(img_show.max()-img_show.min()+1e-6)

        foreground_score = np.array(foreground_score[:,0, ...].data.cpu().numpy() * 255,dtype=np.uint8)
        gt_semantic_seg = np.array(gt_semantic_seg[:,0, ...].data.cpu().numpy() * 255,dtype=np.uint8)
        for i in range(img_show.shape[0]):
            name = '%09d'%random.randint(1,1000)
            img_name = '%s.jpg'%name
            img_c = img_show[i,...,0:3]
            # img_c = imshow_det_bboxes(img=img_c,bboxes=gt_bboxes[i].data.cpu().numpy(),labels=gt_labels[i].data.cpu().numpy(),class_names=['fire','smoke'],show=False)
            img_c = imshow_det_bboxes(img=img_c,bboxes=gt_bboxes[i].data.cpu().numpy(),labels=gt_labels[i].data.cpu().numpy(),class_names=['smoke','fire'],show=False)
            cv2.imwrite('../show_input/'+img_name,img_c[:,:,[2,1,0]])
            pred = foreground_score[i,...]
            pred = cv2.resize(pred,fx=None,fy=None,dsize=(img_c.shape[1],img_c.shape[0]))
            cv2.imwrite('../show_input/' + img_name.replace('.jpg','_seg.jpg'), pred)
            cv2.imwrite('../show_input/' + img_name.replace('.jpg', '_gt.jpg'), gt_semantic_seg[i])
            break

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        logits = self.classification_head(x[-1])
        scores = torch.nn.functional.softmax(logits,dim=-1)
        return scores

    def onnx_export(self, img, img_metas, with_nms=True):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        res = self.bbox_head.onnx_export( *outs, img_metas, with_nms=with_nms)
        return res
