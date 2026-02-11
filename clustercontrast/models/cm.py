import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from .losses import CrossEntropyLabelSmooth
from torch.cuda import amp
# from clustercontrast.losses.focal_loss import FocalLoss
from .hm import cm, cm_hard, cm_avg, cm_hybrid_v2  # , tccl


# From PCLHD
class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)  # 更新Memory用
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors  # 更新Memory用

        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)

        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            # -------------------------------------------------
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

            # -------------------------------------------------
            hard = np.argmin(np.array(distances))
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


# From PCLHD
def pcl_cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.1, mode=' CM',num_instances=32):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.num_instances = num_instances
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

        self.cross_entropy_ls = CrossEntropyLabelSmooth(num_classes=num_samples).cuda()

    def forward(self, inputs , targets, cam=False, use_instance=True):
            inputs = F.normalize(inputs, dim=1).cuda()

            if cam:  # 默认分支
                outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum, self.num_instances)

                # 计算Instance
                out_list = torch.chunk(outputs, self.num_instances + 1, dim=1)  # 33份的切分
                out = torch.stack(out_list[1:], dim=0)  # Instance
                neg = torch.max(out, dim=0)[0]
                pos = torch.min(out, dim=0)[0]
                mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
                logits = mask * pos + (1 - mask) * neg
                loss_instance = self.cross_entropy(logits / self.temp, targets)

                # 计算Centroid
                # loss_centroid = self.cross_entropy(out_list[0] / self.temp, targets)
                loss_centroid = self.cross_entropy(outputs / self.temp, targets)


                return loss_centroid , loss_instance
            elif not use_instance:
                outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum, self.num_instances)

                # 计算Centroid
                # loss_centroid = self.cross_entropy(out_list[0] / self.temp, targets)
                loss_centroid = self.cross_entropy(outputs / self.temp, targets)

                return loss_centroid
            else:
                outputs = cm_hard(inputs, targets, self.features, self.momentum)
                outputs /= self.temp
                loss = self.cross_entropy(outputs, targets)

                return loss



class ClusterMemoryCenter(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.1, mode=' CM',num_instances=32):
        super(ClusterMemoryCenter, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.num_instances = num_instances
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

        # self.register_buffer('features', torch.zeros(2 * num_samples, num_features))  #
        # self.register_buffer('instance_feats', torch.zeros(self.num_instances * num_samples, num_features))  #

        # self.cross_entropy_ls = CrossEntropyLabelSmooth(num_classes=num_samples).cuda()

    def forward(self, inputs , targets):
        # print('xxxxx shape', inputs.shape, targets.shape)

        outputs = pcl_cm_hybrid(inputs,
                                targets,
                                self.features, # 这里为Buffer中的
                                self.momentum)
        outputs /= self.temp  # as tau
        # print('xx out shape', outputs.shape)

        mean, hard = torch.chunk(outputs, 2, dim=1)

        # print('mean hard shape', mean.shape, hard.shape)

        r = 0.2
        loss1 = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
        return loss1

        '''    
        inputs = F.normalize(inputs, dim=1).cuda()

        # ------------------------------------------------------------------------------------------
        outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum, self.num_instances)

        loss_centroid = self.cross_entropy(outputs / self.temp, targets)

        return loss_centroid
        '''


class ClusterMemoryPCLHD(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', hard_weight=0.5, smooth=0,
                 num_instances=16):
        super(ClusterMemoryPCLHD, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_samples, 0.1, True)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()

        # 将Memory注册为Buffer，不参与梯度计算
        # buffer相较于属性：自动处理保存和加载，避免计算梯度
        if self.cm_type == 'CM':
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid':
            self.register_buffer('features', torch.zeros(2 * num_samples, num_features))
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))

    def forward(self, inputs, targets, model_name='encoder'):
        inputs = F.normalize(inputs, dim=1).cuda()

        if self.cm_type == 'CM':
            outputs = cm(inputs,
                         targets,
                         self.features,
                         self.momentum)  # call forward
            outputs /= self.temp

            loss = self.cross_entropy(outputs, targets)  # 这里是对比损失
            return loss

        elif self.cm_type == 'CMhybrid':  # Instance + Cluster
            outputs = cm_hybrid(inputs,
                                targets,
                                self.features,
                                self.momentum)  # Define Forward and Backward.
            outputs /= self.temp

            mean, hard = torch.chunk(outputs, 2, dim=1)
            r = 0.2
            loss = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
            return loss


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors

        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        # print('---------- 更新Hybrid  Memory')
        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            # Mean，更新Features
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean  # 动量更新
            ctx.features[index] /= ctx.features[index].norm()

            # Hard，更新Features
            hard = np.argmin(np.array(distances))
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))