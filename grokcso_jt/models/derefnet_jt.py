import os
from typing import List, Tuple, Dict

import jittor as jt
from jittor import nn
import numpy as np
import scipy.io as sio


class MSELossJT(jt.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def execute(self, img1: jt.Var, img2: jt.Var) -> jt.Var:
        return self.alpha * jt.mean((img1 - img2) ** 2)


class BasicBlockJT(jt.Module):
    """
    BasicBlock with soft thresholding
    """
    def __init__(self):
        super().__init__()

        self.lambda_step = jt.array(0.5)
        self.soft_thr = jt.array(0.01)

        self.conv1_forward = nn.Conv(1, 32, 3, 1, 1)
        self.conv2_forward = nn.Conv(32, 32, 3, 1, 1)
        self.conv1_backward = nn.Conv(32, 32, 3, 1, 1)
        self.conv2_backward = nn.Conv(32, 1, 3, 1, 1)

    def execute(self, x: jt.Var, PhiTPhi: jt.Var, PhiTb: jt.Var) -> Tuple[jt.Var, jt.Var]:
        # Gradient descent step
        x = x - self.lambda_step * (x @ PhiTPhi)
        x = x + self.lambda_step * PhiTb

        t, hw = x.shape
        x_input = x.reshape(t, 1, 33, 33)

        # Forward transform
        x_f = nn.relu(self.conv1_forward(x_input))
        x_f = self.conv2_forward(x_f)

        # Soft-thresholding using ReLU (Jittor compatible)
        # soft(x, θ) = max(x-θ, 0) - max(-x-θ, 0)
        pos = nn.relu(x_f - self.soft_thr)
        neg = nn.relu(-x_f - self.soft_thr)
        x_soft = pos - neg

        # Backward transform
        x_b = nn.relu(self.conv1_backward(x_soft))
        x_b = self.conv2_backward(x_b)
        x_pred = x_b.reshape(t, hw)

        # Symmetry loss
        x_sym = nn.relu(self.conv1_backward(x_f))
        x_sym = self.conv2_backward(x_sym)
        x_sym = x_sym - x_input

        symloss = x_sym

        return x_pred, symloss


class TDFAJT(jt.Module):
    """
    Temporal alignment module with fixed positional encoding
    """
    def __init__(self, position_dim: int = 1089):
        super().__init__()
        self.position_dim = position_dim

        self.mlp = nn.Linear(position_dim, position_dim)
        self.sigmoid = nn.Sigmoid()

        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv(1, 32, 3, 1, 1),
            nn.ReLU(),
        )

        self.tail = nn.Sequential(
            nn.Conv(32 * 5, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv(64, 1, 3, 1, 1),
        )

    def generate_positional_encoding(self, seq_len: int, feature_dim: int) -> jt.Var:
        """Simple positional encoding - Jittor compatible"""
        time_steps = jt.arange(seq_len).reshape(-1, 1).float()
        pe = time_steps.broadcast((seq_len, feature_dim))
        pe = pe / float(seq_len)
        return pe

    def execute(self, x: jt.Var) -> Tuple[List[jt.Var], List[int], Dict[int, jt.Var]]:
        t, hw = x.shape

        # Positional encoding
        pos = self.generate_positional_encoding(t, self.position_dim)
        pos = self.sigmoid(self.mlp(pos))
        x = x * pos

        x_img = x.reshape(t, 1, 33, 33)

        final: List[jt.Var] = []
        index: List[int] = []
        aligned_imgs: Dict[int, jt.Var] = {}

        for j in range(2, t - 2):
            feats = []
            for offset in range(-2, 3):
                m = j + offset
                feat = self.feat_extract(x_img[m:m + 1])
                feats.append(feat)

            feats_cat = jt.concat(feats, dim=1)
            out = self.tail(feats_cat)

            final.append(out.reshape(1, hw))
            aligned_imgs[j] = feats_cat.reshape(5, -1)
            index.append(j)

        return final, index, aligned_imgs


class DeRefNetJT(jt.Module):
    """
    DeRefNet for CSIST unmixing - Jittor version
    """
    def __init__(self, layer_no: int = 9):
        super().__init__()

        # Load measurement matrix and initialization matrix
        phi_path = os.path.join("SeqCSIST", "data", "phi_0.5.mat")
        phi_mat = sio.loadmat(phi_path)["phi"]
        self.Phi = jt.array(phi_mat.astype(np.float32))

        qinit_path = os.path.join("SeqCSIST", "data", "track_5000_20", "train", "qinit.mat")
        qinit_mat = sio.loadmat(qinit_path)["Qinit"]
        self.Qinit = jt.array(qinit_mat.astype(np.float32))

        self.layer_no = layer_no
        self.blocks = nn.ModuleList([BasicBlockJT() for _ in range(layer_no)])
        self.tdfa = TDFAJT(position_dim=1089)
        self.fg_loss = MSELossJT(alpha=1.0)

    def execute(self, batch_x: jt.Var, gt_img_11: jt.Var) -> Dict[str, jt.Var]:
        """
        Args:
            batch_x: [t, 1089] Ground truth
            gt_img_11: [t, 121] Compressed measurements
        Returns:
            dict with losses
        """
        # Precompute
        Phi = self.Phi
        Qinit = self.Qinit
        PhiTPhi = (Phi.transpose(0, 1) @ Phi)
        PhiTb = gt_img_11 @ Phi

        # Initialize
        x = gt_img_11 @ Qinit.transpose(0, 1)

        # Feature extraction through ISTA unfolding
        layers_sym = []
        for i in range(self.layer_no):
            x, sym = self.blocks[i](x, PhiTPhi, PhiTb)
            layers_sym.append(sym)

        # Temporal alignment
        final_list, index_list, aligned = self.tdfa(x)

        # Loss computation
        # 1. Constraint loss
        loss_constraint = jt.array(0.0)
        for sym in layers_sym:
            loss_constraint = loss_constraint + jt.mean(sym ** 2)

        # 2. Alignment loss (simplified)
        loss_align = jt.array(0.0)
        for idx in index_list:
            feats = aligned[idx]
            ref = feats[2:3]
            neigh = jt.concat([feats[0:2], feats[3:5]], dim=0)
            loss_align = loss_align + jt.sum(jt.abs(neigh - ref))

        # 3. Regression loss
        loss_fg = jt.array(0.0)
        for i, idx in enumerate(index_list):
            pred = final_list[i]
            gt = batch_x[idx:idx + 1]
            loss_fg = loss_fg + self.fg_loss(pred, gt)

        # Total loss
        alpha = 0.01
        gamma = 0.01
        loss = gamma * loss_constraint + loss_fg + alpha * loss_align

        return dict(
            loss=loss,
            loss_constraint=gamma * loss_constraint,
            loss_align=alpha * loss_align,
            loss_reg=loss_fg,
        )