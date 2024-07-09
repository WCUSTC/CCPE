import numpy as np
import torch,sys,cv2
import torchvision
sys.path.append(r'D:\mmdetection-master-win\tools\deployment\pytorch2caffemaster')
from tools.deployment.pytorch2caffemaster.pytorch2caffe import pytorch2caffe
from torchvision.models import resnet
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.backbones.swin import SwinTransformer
from mmdet.models.detectors.yolox import YOLOX
from mmdet.models.detectors.yolo import YOLOV3


'''  yolox-s'''
# backbone = {'type': 'CSPDarknet', 'deepen_factor': 0.33, 'widen_factor': 0.5}
# neck = {'type': 'YOLOXPAFPN', 'in_channels': [128, 256, 512], 'out_channels': 128, 'num_csp_blocks': 1}
# bbox_head = {'type': 'YOLOXHead', 'num_classes': 2, 'in_channels': 128, 'feat_channels': 128}
# name = "YOLOX-s"
# model = YOLOX(backbone,neck,bbox_head)
# model.forward = model.caffe_export


"""ResNet"""
# res18 = {'depth': 18, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1,
#          'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch',
#          'init_cfg': {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet18'}}
# name = 'resnet18'
# model = ResNet(**res18)

'''Swin-t'''
# swint = {'embed_dims': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24],
#             'window_size': 7, 'mlp_ratio': 4, 'qkv_bias': True, 'qk_scale': None, 'drop_rate': 0.0,
#             'attn_drop_rate': 0.0, 'drop_path_rate': 0.2, 'patch_norm': True, 'out_indices': (1, 2, 3),
#             'with_cp': False, 'convert_weights': True, 'init_cfg': {'type': 'Pretrained',
#                                                                     'checkpoint': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'}}
# name = 'swin-t'
# model = SwinTransformer(**swint)
# model.forward = model.caffe_export

'''ResNet50 + FPN + YOLOX'''
# yolox_res50 ={'input_size': (640, 640), 'random_size_range': (15, 25), 'random_size_interval': 10, 'backbone': {'type': 'ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': -1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch', 'init_cfg': None}, 'neck': {'type': 'FPN', 'in_channels': [512, 1024, 2048], 'out_channels': 128, 'num_outs': 3}, 'bbox_head': {'type': 'YOLOXHeadSepSample', 'num_classes': 2, 'in_channels': 128, 'feat_channels': 128, 'rate': 200, 'pos_sample': {'type': 'Random', 'rate': 10}, 'neg_sample': {'type': 'OHEM', 'rate': 190, 'mu': -300000, 'mse': 500000}, 'stacked_convs': 2, 'strides': [8, 16, 32], 'use_depthwise': False, 'dcn_on_last_conv': False, 'conv_bias': 'auto', 'conv_cfg': None, 'norm_cfg': {'type': 'BN', 'momentum': 0.03, 'eps': 0.001}, 'act_cfg': {'type': 'Swish'}, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0}, 'loss_bbox': {'type': 'IoULoss', 'mode': 'square', 'eps': 1e-16, 'reduction': 'sum', 'loss_weight': 5.0}, 'loss_obj': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0}, 'loss_l1': {'type': 'L1Loss', 'reduction': 'sum', 'loss_weight': 1.0}}, 'train_cfg': None, 'test_cfg': {'score_thr': 0.01, 'nms': {'type': 'nms', 'iou_threshold': 0.65}}}
# name = "YOLOX-Res50"
# model = YOLOX(**yolox_res50)
# model.forward = model.caffe_export

'''ResNet50 + FPNDeconv + YOLOX'''
# yolox_res50 ={'input_size': (640, 640), 'random_size_range': (15, 25), 'random_size_interval': 10, 'backbone': {'type': 'ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (1, 2, 3), 'frozen_stages': -1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch', 'init_cfg': None}, 'neck': {'type': 'FPNDeconv', 'in_channels': [512, 1024, 2048], 'out_channels': 128, 'num_outs': 3}, 'bbox_head': {'type': 'YOLOXHeadSepSample', 'num_classes': 2, 'in_channels': 128, 'feat_channels': 128, 'rate': 200, 'pos_sample': {'type': 'Random', 'rate': 10}, 'neg_sample': {'type': 'OHEM', 'rate': 190, 'mu': -300000, 'mse': 500000}, 'stacked_convs': 2, 'strides': [8, 16, 32], 'use_depthwise': False, 'dcn_on_last_conv': False, 'conv_bias': 'auto', 'conv_cfg': None, 'norm_cfg': {'type': 'BN', 'momentum': 0.03, 'eps': 0.001}, 'act_cfg': {'type': 'Swish'}, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0}, 'loss_bbox': {'type': 'IoULoss', 'mode': 'square', 'eps': 1e-16, 'reduction': 'sum', 'loss_weight': 5.0}, 'loss_obj': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0}, 'loss_l1': {'type': 'L1Loss', 'reduction': 'sum', 'loss_weight': 1.0}}, 'train_cfg': None, 'test_cfg': {'score_thr': 0.01, 'nms': {'type': 'nms', 'iou_threshold': 0.65}}}
# name = "YOLOX-Res50"
# model = YOLOX(**yolox_res50)
# param_path = r"D:\mmdetection-master-win\work_dirs_wildfire\yolox_res50_FPNDeconv_SepInd10#OHEM190\epoch_40.pth"
# model_dict = torch.load(param_path, map_location=torch.device('cpu'))
# model.load_state_dict(model_dict['state_dict'],strict=False)
# model.forward = model.caffe_export


'''Darknet53 + YOLOV3Neck + yolov3'''
# yolov3 = {'backbone': {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5), 'init_cfg': None}, 'neck': {'type': 'YOLOV3Neck', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 2, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}}, 'train_cfg': None, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}
# name = "yolov3-d53"
# model = YOLOV3(**yolov3)
# param_path = r"D:\mmdetection-master-win\work_dirs_wildfire\yolov3_d53_mstrain-608_273e_coco\epoch_68.pth"
# model_dict = torch.load(param_path, map_location=torch.device('cpu'))
# model.load_state_dict(model_dict['state_dict'],strict=False)
# model.forward = model.caffe_export


'''Darknet53 + YOLOV3NeckDeconv + yolov3'''
# yolov3 = {'backbone': {'type': 'Darknet', 'depth': 53, 'out_indices': (3, 4, 5), 'init_cfg': None}, 'neck': {'type': 'YOLOV3NeckDeconv', 'num_scales': 3, 'in_channels': [1024, 512, 256], 'out_channels': [512, 256, 128]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 2, 'in_channels': [512, 256, 128], 'out_channels': [1024, 512, 256], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}}, 'train_cfg': None, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}
# name = "yolov3-d53Deconv"
# model = YOLOV3(**yolov3)
# param_path = r"D:\mmdetection-master-win\work_dirs_wildfire\yolov3_d53_deconv_mstrain-608_273e_coco\epoch_12.pth"
# model_dict = torch.load(param_path, map_location=torch.device('cpu'))
# model.load_state_dict(model_dict['state_dict'],strict=False)
# model.forward = model.caffe_export

'''mobilenetv2 + YOLOV3NeckDeconv + yolov3'''
yolov3 = {'backbone': {'type': 'MobileNetV2', 'out_indices': (2, 4, 6), 'act_cfg': {'type': 'LeakyReLU', 'negative_slope': 0.1}, 'init_cfg': None}, 'neck': {'type': 'YOLOV3NeckDeconv', 'num_scales': 3, 'in_channels': [320, 96, 32], 'out_channels': [96, 96, 96]}, 'bbox_head': {'type': 'YOLOV3Head', 'num_classes': 2, 'in_channels': [96, 96, 96], 'out_channels': [96, 96, 96], 'anchor_generator': {'type': 'YOLOAnchorGenerator', 'base_sizes': [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]], 'strides': [32, 16, 8]}, 'bbox_coder': {'type': 'YOLOBBoxCoder'}, 'featmap_strides': [32, 16, 8], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_conf': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0, 'reduction': 'sum'}, 'loss_xy': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 2.0, 'reduction': 'sum'}, 'loss_wh': {'type': 'MSELoss', 'loss_weight': 2.0, 'reduction': 'sum'}}, 'train_cfg': None, 'test_cfg': {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}}
name = "yolov3-mobileDeconv"
model = YOLOV3(**yolov3)
param_path = r"D:\mmdetection-master-win\work_dirs_wildfire\yolov3_mobilenetv2_deconv\epoch_48.pth"
model_dict = torch.load(param_path, map_location=torch.device('cpu'))
model.load_state_dict(model_dict['state_dict'],strict=False)
model.forward = model.caffe_export

model.eval()
def SaveDemo():
    dummy_input = torch.ones(size=shape)
    pytorch2caffe.trans_net(model, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))
    pass


if __name__ == '__main__':
    shape = [1,3,544, 544]
    img = cv2.imread(r"E:/fire-data/00shengpingtai/shengpingtai_test/Smoke/JPEGImages/26c6c72dd81e40c8a8a4f7c7a7d85102_2023-04-14---18--55--27.241076--00095.jpg")

    img = cv2.resize(img,dsize=shape[-2:],fx=None,fy=None)
    cv2.imwrite(r'E:\01Projects\HuaWeiHaoWang\RuyiStudio-2.0.41\workspace\V3MoV2Deco3516\IMGresized.jpg',img)
    # img = cv2.imread(r'E:\01Projects\HuaWeiHaoWang\RuyiStudio-2.0.41\workspace\V3MoV2Deco3516\IMGresized.jpg')
    img = img[:,:,(2,1,0)] ##BGR 2 RGB
    img = img.transpose(2,0,1)## H,W C  --> C ,H,W
    img = img[np.newaxis,...]
    img = torch.from_numpy(img).float()
    img = img/255.0
    # SaveDemo()
    pytorch_output = model(img)
    pass
