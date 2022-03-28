import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2





class FEM(nn.Module):
    def __init__(self, reduce_dim=256, pyramid_bins=[60, 30, 15, 8]):
        super(FEM, self).__init__()
        self.pyramid_bins = pyramid_bins
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        mask_add_num = 2
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)        

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, 2, kernel_size=1)
        )      

    def forward(self, query_feat, supp_feat, corr_query_mask):
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        return out