import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import model.resnet as models
import random
import time
import cv2

from model.fem import FEM


'''
Acknowledgement:
The implementation of FewShotSeg is based on that of PANet:
https://github.com/kaixin96/PANet
'''

class FewShotSeg(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None, args=None, backbone='resnet50'):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.args = args

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        if backbone == 'resnet50':
            print('...Using Res-50')
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            print('...Using Res-101')
            resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        core_dim = 2048

        out_channels = 512
        main_dim = 512          

        self.down = nn.Sequential(
            nn.Conv2d(core_dim, out_channels, kernel_size=1, padding=0, bias=True),
        )   
                  
        
        if self.args.use_coco:
            self.main_proto = nn.Parameter(torch.randn(61, main_dim).cuda())
        else:
            self.main_proto = nn.Parameter(torch.randn(16, main_dim).cuda())

        gamma_dim = 1
        self.gamma_conv = nn.Sequential(
            nn.Linear(out_channels*2, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, gamma_dim)
        )  

        self.beta_conv = nn.Sequential(
            nn.Linear(out_channels*2, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 1)
        )              
        self.attn_conv = nn.ModuleList([])
        self.attn_conv.append(nn.Sequential(
            nn.Conv2d(1024+512, 256, 1),
            nn.ReLU(inplace=True),
        ))          

        self.fem_module = FEM(reduce_dim=256, pyramid_bins=[60, 30, 15, 8])                        

    def encoder(self, imgs, masks=None):
        feat0 = self.layer0(imgs)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)                                    
        feat4 = self.layer4(feat3)

        mid_feats = torch.cat([feat3, feat2], 1)
        return self.down(feat4), mid_feats

        

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, s_raw_label, raw_label, it=None):
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        self.n_shots = n_shots

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0) #[way*shot*B+N*B, 3, H, W]

        masks_concat = torch.cat(fore_mask[0], dim=0) 

        img_fts, img_mid_fts = self.encoder(imgs_concat, masks_concat) #[way*shot*B+N*B, nFeat, h, w]
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        supp_mid_fts = img_mid_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_mid_fts = img_mid_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'            
        

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        main_loss = 0
        fake_main_loss = 0
        outputs = []
        pre_outputs = []

        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]], s_raw_label[epi, shot])
                            for shot in range(n_shots)] for way in range(n_ways)]

            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], s_raw_label[epi, shot])  # 
                            for shot in range(n_shots)] 
                            for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### get aligned loss
            new_qry_fts = qry_fts[:, epi]

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(new_qry_fts, prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            pre_outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))  
                    

            ###### Prototype alignment loss ######
            align_loss_epi, new_supp_prototypes, new_meta_supp_prototypes = self.alignLoss(\
                        new_qry_fts, pred, supp_fts[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi], raw_label[epi], iter=it)
            if self.training:
                align_loss += align_loss_epi

        
            if len(new_supp_prototypes) == 0:
                outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))  
            else:
                pred_list = []
                for sidx in range(n_shots):
                    tmp_prototype = [new_supp_prototypes[0][sidx], new_supp_prototypes[1][sidx]]
                    dist = [self.calDist(new_qry_fts, prototype) for prototype in tmp_prototype]
                    tmp_pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'         
                    pred_list.append(tmp_pred)
                pred = sum(pred_list) / len(pred_list)      

                meta_pred_list = []
                outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True))   

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])        
        raw_output = output.clone()     
        pre_output = torch.stack(pre_outputs, dim=1)  # N x B x (1 + Wa) x H x W
        pre_output = pre_output.view(-1, *pre_output.shape[2:]) 

        output = F.interpolate(output, size=qry_mid_fts[0].shape[-2:], mode='bilinear', align_corners=True)
        output = self.get_refined_pred(output, qry_mid_fts, supp_mid_fts, fore_mask) 
        output = F.interpolate(output, size=img_size, mode='bilinear', align_corners=True)             

        main_loss = 0 #torch.zeros(1).cuda()
        if self.training:
            supp_fts = supp_fts.contiguous().view(n_shots, batch_size, -1, *fts_size).permute(1, 0, 2, 3, 4).contiguous().view(batch_size*n_shots, -1, *fts_size)
            supp_raw_labels = s_raw_label.view(batch_size*n_shots, *s_raw_label.size()[2:])
            qry_fts = qry_fts.permute(1, 0, 2, 3, 4).view(batch_size*n_queries, -1, *fts_size)

            new_proto, fake_proto, _, _, _ = self.generate_fake_proto(proto=self.main_proto, x=supp_fts, y=supp_raw_labels)

            x_fake = self.get_pred(torch.cat([supp_fts, qry_fts], 0), new_proto)
            x_fake = F.interpolate(x_fake, size=raw_label.size()[1:], mode='bilinear', align_corners=True)
            fake_main_loss = self.criterion(x_fake, torch.cat([supp_raw_labels, raw_label], 0))

            x = self.get_pred(torch.cat([supp_fts, qry_fts], 0), self.main_proto)
            x = F.interpolate(x, size=raw_label.size()[1:], mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, torch.cat([supp_raw_labels, raw_label], 0))

            main_loss = 0.5 * main_loss + 0.5 * fake_main_loss 
            return output, pre_output, align_loss / batch_size, main_loss                


        return output, align_loss / batch_size


    def get_refined_pred(self, pred, query_mid_feat, supp_mid_feat, mask):
        supp_mid_feat = supp_mid_feat[0]    # k, b, c, h, w
        query_mid_feat = query_mid_feat[0]  # b, c, h, w
        query_mid_feat = self.attn_conv[0](query_mid_feat)
        mask = mask[0]  # k, b, h, w
        _, c, h, w = query_mid_feat.shape[:]
        pred_list = []
        pred = F.softmax(pred, 1)


        supp_vec_list = []
        for k in range(self.n_shots):
            tmp_supp = supp_mid_feat[k] # b, c, h, w
            tmp_supp = self.attn_conv[0](tmp_supp)
            tmp_mask = mask[k]
            tmp_mask = F.interpolate(tmp_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True)    # b, 1, h, w
            supp_vec = torch.sum(tmp_supp*tmp_mask, dim=(2,3)) / (torch.sum(tmp_mask, dim=(2,3)) + 1e-12)
            supp_vec_list.append(supp_vec)
        supp_vec = sum(supp_vec_list) / len(supp_vec_list)
        tmp_pred = self.fem_module(query_feat=query_mid_feat, supp_feat=supp_vec.unsqueeze(-1).unsqueeze(-1), corr_query_mask=pred)
        pred = tmp_pred
        return pred


    def get_pred(self, x, proto):
        # x: [b, c, h, w]
        # proto: [cls, c]
        b, c, h, w = x.size()[:]
        cls_num = proto.size(0)
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)
        proto = proto / (torch.norm(proto, 2, 1, True) + 1e-12)
        x = x.contiguous().view(b, c, h*w)  # b, c, hw
        proto = proto.unsqueeze(0)  # 1, cls, c
        pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 20

    def generate_fake_proto(self, proto, x, y, ratio=None):
        b, c, h, w = x.size()[:]
        tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h,w), mode='nearest')
        unique_y = list(tmp_y.unique())
        raw_unique_y = list(tmp_y.unique())
        if 0 in unique_y:
            unique_y.remove(0)
        if 255 in unique_y:
            unique_y.remove(255)
        all_labels_list = list(range(self.main_proto.size(0)))

        novel_num = len(unique_y) #// 2
        if novel_num == 0:
            novel_num = len(unique_y)
        fake_novel = random.sample(unique_y, novel_num)
        for fn in fake_novel:
            unique_y.remove(fn) # the remaining classes are context
        fake_context = unique_y
        
        new_proto = self.main_proto.clone()
        new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)
        feat_proto_list = []
        ori_proto_list = []
        for fn in fake_novel:
            # directly replace as the novel class
            tmp_mask = (tmp_y == fn).float()
            tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
            fake_vec[fn.long()] = 1
            new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec
            feat_proto_list.append(tmp_feat.unsqueeze(0))  # 1,c
            ori_proto_list.append(self.main_proto[fn.long()].unsqueeze(0))
            all_labels_list.remove(fn)

        fake_proto = new_proto.clone()

        left_proto_list = []
        for tmp_cls in all_labels_list:
            left_proto_list.append(self.main_proto[tmp_cls].unsqueeze(0))

        feat_proto = torch.cat(feat_proto_list, 0)
        ori_proto = torch.cat(ori_proto_list, 0)
        left_proto = torch.cat(left_proto_list, 0)
                
        return new_proto, fake_proto, feat_proto, ori_proto, left_proto



    def getFeatures(self, fts, mask, raw_labels):
        #print(raw_labels.shape) # torch.Size([ 473, 473])
        fts = F.normalize(fts, 2, 1)        
        raw_fts = fts.clone()
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True) # 1, c, h, w
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C


        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask, qry_raw_label, iter=0):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]        
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        new_supp_bg_prototype_list = []
        new_supp_fg_prototype_list = []
        beta_supp_bg_prototype_list = []
        beta_supp_fg_prototype_list = []
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]

            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]] # c, h, w

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Compute Loss
                query_supp_loss = F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
                loss = loss + query_supp_loss

                supp_fg_mask = (supp_label == 1).float()
                tmp_supp_fg_mask = F.interpolate(supp_fg_mask.unsqueeze(0).unsqueeze(0), size=img_fts.shape[-2:], mode='bilinear', align_corners=True)
                supp_fg_proto = torch.sum(img_fts * tmp_supp_fg_mask, dim=(2, 3))
                supp_fg_proto = supp_fg_proto / (tmp_supp_fg_mask.sum((2, 3)) + 1e-5)  # C

                supp_bg_mask = (supp_label == 0).float()
                tmp_supp_bg_mask = F.interpolate(supp_bg_mask.unsqueeze(0).unsqueeze(0), size=img_fts.shape[-2:], mode='bilinear', align_corners=True)
                supp_bg_proto = torch.sum(img_fts * tmp_supp_bg_mask, dim=(2, 3))
                supp_bg_proto = supp_bg_proto / (tmp_supp_bg_mask.sum((2, 3)) + 1e-5)  # C

                supp_proto_list = [supp_bg_proto, supp_fg_proto]
                real_supp_dist = [self.calDist(img_fts, supp_proto) for supp_proto in supp_proto_list]
                real_supp_pred = torch.stack(real_supp_dist, dim=1)
                real_supp_pred = F.interpolate(real_supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)  
                real_supp_loss = F.cross_entropy(real_supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways 

                align_kl_loss = 0
                loss = loss + real_supp_loss

                ### prototype refinement
                query_pred_dist = [self.calDist(qry_fts, prototype) for prototype in supp_proto_list]
                query_pred = torch.stack(query_pred_dist, dim=1)
                b, n, h, w = query_pred.shape[:]
                real_query_pred = query_pred.view(b, n, h*w)

                real_query_pred = F.softmax(real_query_pred, -1)    # b, n, hw
                qry_fts_rep = qry_fts.view(b, -1, h*w)
                real_query_proto = torch.bmm(real_query_pred, qry_fts_rep.permute(0, 2, 1)) # b, n, c
                real_supp_proto = torch.cat([supp_bg_proto.unsqueeze(0), supp_fg_proto.unsqueeze(0)], 0).view(1, n, qry_fts.shape[1])
                pred_proto_norm = F.normalize(real_query_proto, 2, -1)
                proto_norm = F.normalize(real_supp_proto, 2, -1)
                pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)   # b, n, 1
                assert pred_weight.shape[0] == 1, 'invalid shape of pred_weight: {}'.format(pred_weight.shape)
                pred_weight = pred_weight * (pred_weight > 0).float()       # b, n, 1    
                pred_weight = pred_weight[0] 
                
                if iter % 20 == 0:
                    print('cosine | fg: {}, bg: {}'.format(pred_weight[0], pred_weight[1]))                        
                # print('supp_proto_list[0]: ', supp_proto_list[0].shape) # [1, 512]
                new_supp_bg_prototype = (1 - pred_weight[0]) * supp_proto_list[0] + pred_weight[0] * prototypes[0]
                new_supp_fg_prototype = (1 - pred_weight[1]) * supp_proto_list[1] + pred_weight[1] * prototypes[1]


                beta_fg_weight = F.sigmoid(self.beta_conv(torch.cat([prototypes[1], supp_proto_list[1]], 1)))
                beta_bg_weight = F.sigmoid(self.beta_conv(torch.cat([prototypes[0], supp_proto_list[0]], 1)))

                if iter % 20 == 0:
                    print('meta | fg: {}, bg: {}'.format(beta_fg_weight, beta_bg_weight))
                beta_supp_bg_prototype = (1 - beta_bg_weight) * supp_proto_list[0] + beta_bg_weight * prototypes[0]
                beta_supp_fg_prototype = (1 - beta_fg_weight) * supp_proto_list[1] + beta_fg_weight * prototypes[1]

                new_supp_bg_prototype = new_supp_bg_prototype + beta_supp_bg_prototype
                new_supp_fg_prototype = new_supp_fg_prototype + beta_supp_fg_prototype
                beta_supp_bg_prototype_list.append(beta_supp_bg_prototype)
                beta_supp_fg_prototype_list.append(beta_supp_fg_prototype)                        

                new_supp_bg_prototype_list.append(new_supp_bg_prototype)
                new_supp_fg_prototype_list.append(new_supp_fg_prototype)

        if len(new_supp_bg_prototype_list) > 0:
            return loss, [new_supp_bg_prototype_list, new_supp_fg_prototype_list], [beta_supp_bg_prototype_list, beta_supp_fg_prototype_list]
        else:
            return loss, [], []                


    def calDist(self, fts, prototype, scaler=20):
        scaler = 20
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist




