"""方案简介（Semi-MeanTeacher-Text-v1）：
work！
使用 Mean Teacher 进行半监督分割训练，监督项为 CE + Dice，无监督项为学生/教师预测一致性约束。
额外引入类别文本特征，与学生端高熵区域聚合后的特征做特征级 InfoNCE 对齐，教师预测提供伪标签与置信度筛选。
该策略要求模型为 ResNetUNet_proto，并依赖 task 对应的文本特征文件。
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseTrainingStrategy

class Endovis2017TextFeatureStore:
    _cache={}
    def __init__(self,args):
        self.base_dir=args.root_path
        self.task=int(args.task)
        self.task_name=f'task{self.task}'
        key=(self.base_dir,self.task)
        if key in Endovis2017TextFeatureStore._cache:self.items=Endovis2017TextFeatureStore._cache[key]
        else:
            with open(os.path.join(self.base_dir,f'{self.task_name}.json'),'r',encoding='utf-8') as f:cfg=json.load(f)
            text_dir=os.path.join(self.base_dir,'data',f'{self.task_name}_text_feature')
            items=[]
            for c in sorted(cfg['classes'],key=lambda x:int(x['label_id'])):
                name=str(c['name'])
                p=os.path.join(text_dir,f"{name.replace(' ','_')}.npy")
                if not os.path.exists(p):raise FileNotFoundError(p)
                items.append({'label_id':int(c['label_id']),'class_name':name,'feature':np.load(p)})
            self.items=items
            Endovis2017TextFeatureStore._cache[key]=self.items
        self.feature_dims=[tuple(x['feature'].shape) for x in self.items]
        self.feature_dim_set=sorted(list({int(x['feature'].reshape(-1).shape[0]) for x in self.items}))

class MeanTeacherTextV1Strategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()
        self.consistency_start_iters=int(args.consistency_start_iters)
        self.num_classes=int(args.num_classes)
        self.high_entropy_ratio=0.3
        self.text_temp=0.07
        self.text_contrast_weight=1
        self.text_low_entropy_weight=1.0
        self.text_high_entropy_weight=0.3
        self.text_feat_align_weight=1.0
        self.text_teacher_conf_thresh=0.6
        if type(self.model).__name__!='ResNetUNet_proto':raise ValueError(f"semi_mean_teacher_text_v1 requires ResNetUNet_proto, got {type(self.model).__name__}")
        self.text_feature_store=Endovis2017TextFeatureStore(args)
        self.text_bank=self._build_text_bank().to(self.device)
        self.text_valid_class_ids=[int(x['label_id']) for x in self.text_feature_store.items if int(x['label_id'])<self.num_classes]

    def _build_text_bank(self):
        dim=int(self.text_feature_store.items[0]['feature'].reshape(-1).shape[0]) if len(self.text_feature_store.items)>0 else 0
        bank=torch.zeros((self.num_classes,dim),dtype=torch.float32)
        for x in self.text_feature_store.items:
            y=torch.from_numpy(x['feature']).float().reshape(-1)
            if int(x['label_id'])<self.num_classes and y.shape[0]==dim:bank[int(x['label_id'])]=y
        return bank

    def _extract_logits_and_feat(self,output):
        if isinstance(output,(tuple,list)):return output[0],output[1] if len(output)>1 else None
        return output,None

    def _compute_info_loss(self,student_feat,student_prob,teacher_prob,text_bank):
        if student_feat is None or student_prob.numel()==0 or teacher_prob.numel()==0:return torch.tensor(0.0,device=self.device)
        text_norm=F.normalize(text_bank,p=2,dim=1)
        pseudo_cls=teacher_prob.argmax(dim=1)
        teacher_conf=teacher_prob.max(dim=1).values>=self.text_teacher_conf_thresh
        ent=-(student_prob*torch.log(student_prob.clamp(min=1e-8))).sum(dim=1)
        thr=torch.quantile(ent.reshape(-1),1.0-self.high_entropy_ratio)
        high_mask=(ent>=thr)&teacher_conf
        if int(high_mask.sum().item())==0:return torch.tensor(0.0,device=self.device)
        feat_h,feat_w=student_feat.shape[2],student_feat.shape[3]
        region_mask=F.interpolate((high_mask&(pseudo_cls<self.num_classes)).unsqueeze(1).float(),size=(feat_h,feat_w),mode='nearest').squeeze(1)>0
        pseudo_cls_feat=F.interpolate(pseudo_cls.unsqueeze(1).float(),size=(feat_h,feat_w),mode='nearest').squeeze(1).long()
        feat_flat=student_feat.permute(0,2,3,1).reshape(-1,student_feat.shape[1])
        mask_flat=region_mask.reshape(-1)
        cls_flat=pseudo_cls_feat.reshape(-1)
        if int(mask_flat.sum().item())==0:return torch.tensor(0.0,device=self.device)
        feat_flat=feat_flat[mask_flat]
        cls_flat=cls_flat[mask_flat]
        pooled_feat=[]
        pooled_cls=[]
        for cls_id in cls_flat.unique(sorted=True):
            cls_mask=cls_flat==cls_id
            if int(cls_mask.sum().item())==0:continue
            pooled_feat.append(feat_flat[cls_mask].mean(dim=0,keepdim=True))
            pooled_cls.append(cls_id.view(1))
        if len(pooled_feat)==0:return torch.tensor(0.0,device=self.device)
        pooled_feat=F.normalize(torch.cat(pooled_feat,dim=0),p=2,dim=1)
        pooled_cls=torch.cat(pooled_cls,dim=0).long()
        logits=torch.matmul(pooled_feat,text_norm.t())/self.text_temp
        log_prob=F.log_softmax(logits,dim=1)
        return (-log_prob.gather(1,pooled_cls.unsqueeze(1)).squeeze(1)).mean()

    def compute_loss(self,batch_data,iter_num=0,epoch=0):
        volume=batch_data['image'].to(self.device)
        depth_tensor=self._get_depth_tensor(batch_data)
        if depth_tensor is not None:volume=torch.cat([volume,depth_tensor],dim=1)
        label=batch_data['label'].to(self.device)
        stud_volume=self._add_noise(volume,strong_flag='s',unlabeled_only=True)
        student_logits,student_feat=self._extract_logits_and_feat(self.model(stud_volume))
        student_prob=torch.softmax(student_logits,dim=1)
        unlabeled_volume=volume[self.labeled_bs:]
        if unlabeled_volume.shape[0]>0:
            with torch.no_grad():
                ema_inputs=self._add_noise(unlabeled_volume,strong_flag='t')
                teacher_logits,_=self._extract_logits_and_feat(self.ema_model(ema_inputs))
                teacher_prob=torch.softmax(teacher_logits,dim=1)
        else:teacher_prob=student_prob.new_empty((0,student_prob.shape[1],student_prob.shape[2],student_prob.shape[3]))
        batch_data['teacher_pred']=teacher_prob
        loss_ce=self.ce_loss(student_logits[:self.labeled_bs],label[:self.labeled_bs].long())
        loss_dice=self.dice_loss(student_prob[:self.labeled_bs],label[:self.labeled_bs].unsqueeze(1))
        supervised_loss=0.5*(loss_dice+loss_ce)

        consistency_weight=self._get_consistency_weight(iter_num)
        teacher_loss=torch.tensor(0.0,device=self.device)
        info_loss=torch.tensor(0.0,device=self.device)
        if iter_num>=self.consistency_start_iters and unlabeled_volume.shape[0]>0:
            student_unlabeled_prob=student_prob[self.labeled_bs:]
            teacher_loss=torch.mean((student_unlabeled_prob-teacher_prob)**2)
            if student_feat is not None:
                student_unlabeled_feat=student_feat[self.labeled_bs:]
                info_loss=self._compute_info_loss(student_unlabeled_feat,student_unlabeled_prob,teacher_prob,self.text_bank)
        consistency_loss=teacher_loss+info_loss * 0.5
        total_loss=supervised_loss+consistency_weight*consistency_loss
        return {'total':total_loss,'ce':loss_ce,'dice':loss_dice,'teacher_consistency':teacher_loss,'info_contrast':info_loss,'consistency':consistency_loss,'consistency_weight':consistency_weight}

    def training_step(self,batch_data,iter_num,epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        loss_dict=self.compute_loss(batch_data,iter_num,epoch)
        self._backward_and_step(loss_dict['total'],optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict
