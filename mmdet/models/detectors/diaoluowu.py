from ..builder import DETECTORS

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck



@DETECTORS.register_module()
class HumidityRegressionNR(BaseDetector):
    '''

    '''
    def __init__(self, backbone,neck=None,train_cfg=None,test_cfg=None,):
        super(HumidityRegressionNR, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=128,bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128,out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,rgb,nr,thermal,gt_labels):
        img = nr.repeat(1,3,1,1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.mse_loss(input=res,target=gt_labels)
        losses={"losses_reg":loss}
        return losses

    # def forward(self,rgb,nr,thermal,gt_labels):
    #     img = nr.repeat(1,3,1,1)
    #     x = self.extract_feat(img)[-1]
    #     x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
    #     res = self.outputss(x)
    #     return res
    def forward(self,rgb,nr,thermal,gt_labels, return_loss=True,rescale=True):

        if return_loss:
            return self.forward_train(rgb,nr,thermal,gt_labels)
        else:
            return self.forward_test(rgb,nr,thermal,gt_labels)

    def forward_test(self,rgb,nr,thermal,gt_labels):
        img = nr.repeat(1,3,1,1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def simple_test(self,rgb,nr,thermal,gt_labels):
        img = nr.repeat(1,3,1,1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def aug_test(self,rgb,nr,thermal,gt_labels):
        img = nr.repeat(1,3,1,1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res


@DETECTORS.register_module()
class HumidityRegressionRGB(BaseDetector):
    '''
    '''
    def __init__(self, backbone,neck=None,train_cfg=None,test_cfg=None,):
        super(HumidityRegressionRGB, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=128,bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128,out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,rgb,nr,thermal,gt_labels):
        img = rgb
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.mse_loss(input=res,target=gt_labels)
        losses={"losses_reg":loss}
        return losses


    def forward(self,rgb,nr,thermal,gt_labels, return_loss=True,rescale=True):

        if return_loss:
            return self.forward_train(rgb,nr,thermal,gt_labels)
        else:
            return self.forward_test(rgb,nr,thermal,gt_labels)

    def forward_test(self,rgb,nr,thermal,gt_labels):
        img = rgb
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def simple_test(self,rgb,nr,thermal,gt_labels):
        img = rgb
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def aug_test(self,rgb,nr,thermal,gt_labels):
        img = rgb
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res


@DETECTORS.register_module()
class HumidityRegressionThermal(BaseDetector):
    '''
    '''
    def __init__(self, backbone,neck=None,train_cfg=None,test_cfg=None,):
        super(HumidityRegressionThermal, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=128,bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128,out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,rgb,nr,thermal,gt_labels):
        img = thermal
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.l1_loss(input=res[...,0],target=gt_labels)
        losses={"losses_reg":loss}
        return losses


    def forward(self,rgb,nr,thermal,gt_labels, return_loss=True,rescale=True):

        if return_loss:
            return self.forward_train(rgb,nr,thermal,gt_labels)
        else:
            return self.forward_test(rgb,nr,thermal,gt_labels)

    def forward_test(self,rgb,nr,thermal,gt_labels):
        img = thermal
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def simple_test(self,rgb,nr,thermal,gt_labels):
        img = thermal
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def aug_test(self,rgb,nr,thermal,gt_labels):
        img = thermal
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res

@DETECTORS.register_module()
class HumidityRegressionRGB_NR_Thermal(BaseDetector):
    '''
    '''
    def __init__(self, backbone,neck=None,train_cfg=None,test_cfg=None,):
        super(HumidityRegressionRGB_NR_Thermal, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=128,bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128,out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,rgb,nr,thermal,gt_labels):
        img = torch.cat([rgb,nr,thermal],dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.l1_loss(input=res[...,0],target=gt_labels)
        losses={"losses_reg":loss}
        return losses


    def forward(self,rgb,nr,thermal,gt_labels, return_loss=True,rescale=True):

        if return_loss:
            return self.forward_train(rgb,nr,thermal,gt_labels)
        else:
            return self.forward_test(rgb,nr,thermal,gt_labels)

    def forward_test(self,rgb,nr,thermal,gt_labels):
        img = torch.cat([rgb,nr,thermal],dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def simple_test(self,rgb,nr,thermal,gt_labels):
        img = torch.cat([rgb,nr,thermal],dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res
    def aug_test(self,rgb,nr,thermal,gt_labels):
        img = torch.cat([rgb,nr,thermal],dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x,dim=2)[0],dim=2)[0]
        res = self.outputss(x)
        return res

@DETECTORS.register_module()
class HumidityRegressionRGB_NR_Thermal_MVI(BaseDetector):
    def __init__(self, backbone, neck=None, train_cfg=None, test_cfg=None, ):
        super(HumidityRegressionRGB_NR_Thermal_MVI, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=128, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128, out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, thermal,MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.l1_loss(input=res[..., 0], target=gt_labels)
        losses = {"losses_reg": loss}
        return losses

    def forward(self, rgb, nr, thermal, gt_labels, return_loss=True, rescale=True):

        if return_loss:
            return self.forward_train(rgb, nr, thermal, gt_labels)
        else:
            return self.forward_test(rgb, nr, thermal, gt_labels)

    def forward_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, thermal,MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res

    def simple_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, thermal,MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res

    def aug_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, thermal,MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res



@DETECTORS.register_module()
class HumidityRegressionRGB_NR_MVI(BaseDetector):
    def __init__(self, backbone, neck=None, train_cfg=None, test_cfg=None, ):
        super(HumidityRegressionRGB_NR_MVI, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.outputss = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=128, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=128, out_features=1)
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        loss = torch.nn.functional.l1_loss(input=res[..., 0], target=gt_labels)
        losses = {"losses_reg": loss}
        return losses

    def forward(self, rgb, nr, thermal, gt_labels, return_loss=True, rescale=True):

        if return_loss:
            return self.forward_train(rgb, nr, thermal, gt_labels)
        else:
            return self.forward_test(rgb, nr, thermal, gt_labels)

    def forward_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res

    def simple_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res

    def aug_test(self, rgb, nr, thermal, gt_labels):
        MVI = (nr - rgb[:,2:,...])/(nr + rgb[:,2:,...] + 0.0001) + 0.5
        MVI = torch.clip(MVI,min=0)
        MVI = torch.sqrt(MVI)
        img = torch.cat([rgb, nr, MVI], dim=1)
        x = self.extract_feat(img)[-1]
        x = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        res = self.outputss(x)
        return res