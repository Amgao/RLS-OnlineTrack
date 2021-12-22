import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
import math



class DiMPSteepestDescentGN(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'."""

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 score_act='relu', act_param=None, min_filter_reg=1e-3, mask_act='sigmoid',
                 detach_length=float('Inf')):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.Pc = 0
        self.num = 0

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        # Module that predicts the target label function (y in the paper)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        # Module that predicts the target mask (m in the paper)
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias

        # Module that predicts the residual weights (v in the paper)
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # The score actvation and its derivative
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')


    def forward(self, weights, feat, bb, indicator = False, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)
        if not indicator:
            sc = torch.tensor(10.0, requires_grad=False).cuda().half()
            self.Pc = torch.eye(weights.shape[-2] * weights.shape[-1] * feat.shape[1], requires_grad=False).type(torch.cuda.HalfTensor)
            self.Pc.mul_(sc)
            #self.Pc = np.eye(weights.shape[-2] * weights.shape[-1] * feat.shape[1], dtype = "float32") * 10.0
            #self.Pc = torch.autograd.Variable(torch.eye(weights.shape[-2] * weights.shape[-1] * feat.shape[1]).type(torch.cuda.FloatTensor), volatile=True) * 10.0
            self.num = 0

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).view(-1, 2).flip((1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).view(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).view(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).view(num_images, num_sequences, *dist_map.shape[-2:])

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
            b = num_images
        elif isinstance(sample_weight, torch.Tensor):
            b = torch.nonzero(sample_weight.data).size(0)
            sample_weight = sample_weight.sqrt().view(num_images, num_sequences, 1, 1) * spatial_weight

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)

            if compute_losses:
                losses.append(((residuals**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

            # Compute gradient
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + \
                          reg_weight * weights
            weights_grad_RLS = weights_grad.clone()
            def UpdateP(P, feat, weights, score_mask, sample_weight, b):
                feat2 = filter_layer.apply_filter2(feat, weights).detach()
                _, _, H, W = feat2.shape
                FF, _, HH, WW = weights.shape
                S = 2  # stride
                Ho = int(1 + (H - HH) / S)
                Wo = int(1 + (W - WW) / S)
                feat_unf = F.unfold(feat2, (HH, WW), stride = S)
                sample_weight_unf = F.unfold(score_mask * sample_weight, (1, 1), stride = S).detach()
                #feat_unf1 = sample_weight_unf * feat_unf
                #feat_unf = feat_unf * sample_weight_unf
                feat_unf.mul_(sample_weight_unf)
                r = feat_unf.sum(0).sum(1) / math.sqrt(Ho * Wo * b)
                r = r.view(1, -1).half()
                k = torch.mm(P, torch.t(r))
                kk = torch.mm(k.data, torch.t(k.data))
                kk.div_(torch.tensor(1.0, requires_grad=False).cuda().half().data + torch.mm(r.data, k.data))
                #ktest = torch.mm(k.data, torch.t(k.data)) / (1 + torch.mm(r.data, k.data))
                P.sub_(kk)
                #P.sub_(torch.mm(k.data, torch.t(k.data)) / (1 + torch.mm(r.data, k.data)))

            if indicator:
                if i==0:
                    self.num += 1
                    UpdateP(self.Pc, feat, weights, score_mask, sample_weight, b)
                    FF, _, HH, WW = weights.shape
                    if self.num % 100 == 0:
                        sc = torch.tensor(10.0, requires_grad=False).cuda().half()
                        self.Pc = torch.eye(weights.shape[-2] * weights.shape[-1] * feat.shape[1],
                                            requires_grad=False).type(torch.cuda.HalfTensor)
                        self.Pc.mul_(sc)
                    weights_grad_RLS.data = torch.mm(weights_grad.data.view(FF, -1).half(), self.Pc.data).view_as(
                        weights_grad).float()
                    #weights_grad_RLS.data = torch.mm(weights_grad.data.view(FF, -1), self.Pc.float().data).view_as(
                    #    weights_grad)
                else:
                    FF, _, HH, WW = weights.shape
                    #weights_grad_RLS.data = torch.mm(weights_grad.data.view(FF, -1), self.Pc.float().data).view_as(
                    #    weights_grad)
                    weights_grad_RLS.data = torch.mm(weights_grad.data.view(FF, -1).half(), self.Pc.data).view_as(
                        weights_grad).float()
            else:
                # Map the gradient with the Jacobian
                scores_grad = filter_layer.apply_filter(feat, weights_grad)
                scores_grad = sample_weight * (score_mask * scores_grad)

                # Compute optimal step length
                alpha_num = (weights_grad * weights_grad).view(num_sequences, -1).sum(dim=1)
                alpha_den = ((scores_grad * scores_grad).view(num_images, num_sequences, -1).sum(dim=(0,2)) + reg_weight * alpha_num).clamp(1e-8)
                alpha = alpha_num / alpha_den

            # Update filter
            if not indicator:
                weights = weights - (step_length_factor * alpha.view(-1, 1, 1, 1)) * weights_grad
            else:
                alpha = 0.003    # 0.003 for long-term tracking, 0.03 for short-term tracking, 0.06 for VOT
                weights = weights - alpha * weights_grad_RLS

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = self.score_activation(scores, target_mask)
            losses.append((((sample_weight * (scores - label_map))**2).sum() + reg_weight * (weights**2).sum())/num_sequences)

        return weights, weight_iterates, losses
