import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_distance(x, y):
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)
    dist = torch.sum((x - y) ** 2, dim=-1)
    cd = torch.mean(torch.min(dist, dim=2)[0]) + \
         torch.mean(torch.min(dist, dim=1)[0])
    return cd


class ShapeCompletion(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,feat_dim,1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(feat_dim,128,1),
            nn.ReLU(),
            nn.Conv1d(128,64,1),
            nn.ReLU(),
            nn.Conv1d(64,3,1)
        )

    def forward(self, x):
        x = x.transpose(1,2)
        feat = self.encoder(x)
        global_feat = torch.max(feat,2,keepdim=True)[0]
        feat = feat + global_feat
        out = self.decoder(feat)
        return out.transpose(1,2)

class FeatureEncoder(nn.Module):

    def __init__(self, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128,feat_dim,1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(1,2)
        feat = self.net(x)
        return feat.transpose(1,2)


class FTL(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.fc = nn.Linear(16, feat_dim)
    def forward(self, feat, proj):
        B,N,C = feat.shape
        proj = proj.view(B,-1)
        transform = self.fc(proj).unsqueeze(1)
        feat = feat + transform
        return feat


def compute_cost(F1, F2):
    F1 = F.normalize(F1, dim=-1)
    F2 = F.normalize(F2, dim=-1)
    cost = 1 - torch.matmul(F1, F2.transpose(1,2))
    return cost



def sinkhorn(log_alpha, n_iters=10):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)


def estimate_pose(p, q):
    centroid_p = p.mean(dim=1, keepdim=True)
    centroid_q = q.mean(dim=1, keepdim=True)

    p_center = p - centroid_p
    q_center = q - centroid_q

    H = torch.matmul(p_center.transpose(1,2), q_center)
    U,S,V = torch.svd(H)
    R = torch.matmul(V, U.transpose(1,2))
    t = centroid_q - torch.matmul(R, centroid_p.transpose(1,2)).transpose(1,2)
    return R, t



def project_3d_to_2d(points, K):
    x = points[:,:,0]
    y = points[:,:,1]
    z = points[:,:,2]

    u = K[:,0,0]*x/z + K[:,0,2]
    v = K[:,1,1]*y/z + K[:,1,2]
    return torch.stack([u,v], dim=-1)




def bbox_from_points(points2d):
    x_min = torch.min(points2d[:,:,0],dim=1)[0]
    x_max = torch.max(points2d[:,:,0],dim=1)[0]
    y_min = torch.min(points2d[:,:,1],dim=1)[0]
    y_max = torch.max(points2d[:,:,1],dim=1)[0]
    bbox = torch.stack([x_min,y_min,x_max,y_max],dim=1)
    return bbox


def giou_loss(pred, target):

    x1 = torch.max(pred[:,0], target[:,0])
    y1 = torch.max(pred[:,1], target[:,1])
    x2 = torch.min(pred[:,2], target[:,2])
    y2 = torch.min(pred[:,3], target[:,3])

    inter = torch.clamp(x2-x1, min=0) * torch.clamp(y2-y1, min=0)
    area_p = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    area_t = (target[:,2]-target[:,0])*(target[:,3]-target[:,1])
    union = area_p + area_t - inter
    iou = inter/(union+1e-7)
    return 1-iou


class Aerial6D(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.shape_net = ShapeCompletion()
        self.encoder = FeatureEncoder(feat_dim)
        self.ftl = FTL(feat_dim)

    def forward(self, cloud_t, cloud_prev, proj):
        X_t = self.shape_net(cloud_t)
        X_prev = self.shape_net(cloud_prev)
        F_t = self.encoder(X_t)
        F_prev = self.encoder(X_prev)
        # FTL transform
        F_t = self.ftl(F_t, proj)
        F_prev = self.ftl(F_prev, proj)
        # OT matching
        cost = compute_cost(F_t, F_prev)
        M = sinkhorn(-cost)
        # Correspondence
        q = torch.matmul(M, X_prev)
        # Pose 
        R, t = estimate_pose(X_t, q)
        return R, t, M, X_t, X_prev


# loss

class Aerial6DLoss(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self,
                X_t,
                X_prev,
                R_pred,
                t_pred,
                R_gt,
                t_gt,
                bbox_pred,
                bbox_gt):

        pose_pred = torch.matmul(X_t, R_pred.transpose(1,2)) + t_pred
        pose_gt = torch.matmul(X_t, R_gt.transpose(1,2)) + t_gt
        L_pose = torch.mean(torch.abs(pose_pred - pose_gt))
        L_shape = chamfer_distance(X_t, X_prev)
        L_proj = F.l1_loss(bbox_pred, bbox_gt)
        L_giou = torch.mean(giou_loss(bbox_pred, bbox_gt))
        loss = L_pose + L_shape + L_proj + L_giou
        return loss