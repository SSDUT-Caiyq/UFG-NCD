import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from models.resnet50 import modified_resnet50


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_prototypes,
        num_heads,
        num_hidden_layers=1,
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList([Prototypes(output_dim, num_prototypes) for _ in range(num_heads)])
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiPatternHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_prototypes,
        num_pattern,
        num_mlp_layers,
    ):
        super().__init__()
        self.num_pattern = num_pattern

        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_mlp_layers) for _ in range(num_pattern)]
        )

        self.prototypes = torch.nn.ModuleList([Prototypes(output_dim, num_prototypes) for _ in range(num_pattern)])
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward(self, feats, pattern_idx):
        z = self.projectors[pattern_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[pattern_idx](z), z


class MultiProxyModel(nn.Module):
    def __init__(
        self,
        model,
        num_old_classes,
        num_new_classes,
        num_proxy_base=2,
        num_proxy_hard=None,
        num_proxy_local=None,
        num_mlp_layers=1,
        hidden_dim=2048,
        mlp_out_dim=256,
        mode="supervised",
        cgroups=128,
    ):
        super().__init__()
        if model == "resnet50":
            self.encoder = modified_resnet50(pretrained=True)
            self.feat_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = model
            self.feat_dim = 768

        self.num_proxy_base = num_proxy_base
        self.num_proxy_hard = num_proxy_hard
        self.num_proxy_local = num_proxy_local
        self.num_old_classes = num_old_classes
        self.num_new_classes = num_new_classes
        self.mlp_out_dim = mlp_out_dim
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.num_classes = num_old_classes if mode == "supervised" else num_old_classes + num_new_classes
        self.cgroups = cgroups

        self.Proxies_old_base = nn.Parameter(torch.Tensor(self.feat_dim, num_old_classes * num_proxy_base))
        self.Proxies_old_base_label = torch.tensor(np.repeat(np.arange(num_old_classes), num_proxy_base))
        self.Proxies_old_base_label_one_hot = self.to_one_hot(self.Proxies_old_base_label, n_dims=num_old_classes)
        nn.init.kaiming_uniform_(self.Proxies_old_base, a=math.sqrt(5))

        self.Proxies_new_base = nn.Parameter(torch.Tensor(self.feat_dim, num_new_classes * num_proxy_base))
        self.Proxies_new_base_label = (
            torch.tensor(np.repeat(np.arange(num_new_classes), num_proxy_base)) + num_old_classes
        )
        self.Proxies_new_base_label_one_hot = self.to_one_hot(
            self.Proxies_new_base_label, n_dims=num_old_classes + num_new_classes
        )
        nn.init.kaiming_uniform_(self.Proxies_new_base, a=math.sqrt(5))

        # self.Proxies_local = nn.Parameter(torch.Tensor(14 * 14, self.feat_dim))
        # self.Proxies_local_label = torch.tensor(np.append(np.repeat(np.arange(31), 20), np.repeat(np.arange(31, 99), 21)))
        # self.Proxies_local_label_one_hot = self.to_one_hot(self.Proxies_local_label, n_dims=num_old_classes)
        # nn.init.kaiming_uniform_(self.Proxies_local, a=math.sqrt(5))
        # if self.num_proxy_hard:
        #     self.Proxies_old_hard = nn.Parameter(torch.Tensor(self.feat_dim, num_old_classes * num_proxy_hard))
        #     self.Proxies_old_hard_label = torch.tensor(np.repeat(np.arange(num_old_classes), num_proxy_hard))
        #     self.Proxies_old_hard_label_one_hot = self.to_one_hot(self.Proxies_old_hard_label, n_dims=num_old_classes)
        #     nn.init.kaiming_uniform_(self.Proxies_old_hard, a=math.sqrt(5))
        # if self.num_proxy_hard:
        #     self.Proxies_new_hard = nn.Parameter(torch.Tensor(self.feat_dim, num_new_classes * num_proxy_hard))
        #     self.Proxies_new_hard_label = torch.tensor(np.repeat(np.arange(num_new_classes), num_proxy_hard))
        #     self.Proxies_new_hard_label_one_hot = self.to_one_hot(self.Proxies_new_hard_label, n_dims=num_old_classes)
        #     nn.init.kaiming_uniform_(self.Proxies_new_hard, a=math.sqrt(5))

        self.conv_local = nn.Conv2d(in_channels=self.feat_dim, out_channels=self.feat_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(num_features=self.feat_dim)
        self.adapter_old = MLP(
            input_dim=self.feat_dim,
            hidden_dim=5 * self.feat_dim,
            output_dim=self.feat_dim,
            num_hidden_layers=3,
        )

        self.head_region = nn.Linear(in_features=196, out_features=self.cgroups)
        self.label_region = torch.tensor(np.repeat(np.arange(self.cgroups), 1))
        # self.adapter_new = MLP(input_dim=self.feat_dim, hidden_dim=5 * self.feat_dim, output_dim=self.feat_dim, num_hidden_layers=3)
        # if self.num_proxy_local:
        #     self.local_channel_mask_generator = torch.nn.ModuleList([
        #         MLP(input_dim=self.feat_dim, hidden_dim=5 * self.feat_dim, output_dim=self.feat_dim, num_hidden_layers=3)
        #         for _ in range(self.num_proxy_local)])

    def to_one_hot(self, label, n_dims):
        label_ = label.type(torch.LongTensor).view(-1, 1)
        # label_one_hot = torch.zeros(label.size()[0], n_dims).scatter_(1, label, 1).view(*label.shaoe, -1)
        return torch.zeros(label_.size()[0], n_dims).scatter_(1, label_, 1).view(*label.shape, -1)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_pattern.normalize_prototypes()
        self.head_class.normalize_prototypes()

    def forward_proxies(self, feats, feats_map):
        Proxies_old_base_norm = F.normalize(self.Proxies_old_base, p=2, dim=0)
        # Proxies_local_norm = F.normalize(self.Proxies_local, p=2, dim=0)
        if self.num_proxy_hard:
            Proxies_old_hard_norm = F.normalize(self.Proxies_old_hard, p=2, dim=0)
        # Proxies_new_base_norm = F.normalize(self.Proxies_new_base, p=2, dim=0)
        if self.num_proxy_hard:
            Proxies_new_hard_norm = F.normalize(self.Proxies_new_hard, p=2, dim=0)
        if isinstance(feats, list):
            feats = torch.cat(feats, dim=0)
            # feats = feats.detach()
            feats_map = torch.cat(feats_map, dim=0)
        # feats_map = torch.flatten(feats_map, 2, -1)
        # feats_map = F.normalize(feats_map, p=2, dim=-1)
        # Similarity proxy
        # similarity_proxy_old = Proxies_old_norm.t().matmul(Proxies_old_norm)
        # similarity_proxy_new = Proxies_new_norm.t().matmul(Proxies_new_norm)

        # W/O Adapter
        # feats = F.normalize(feats, p=2, dim=1)
        # similarity_old = feats.matmul(Proxies_old_norm)
        # similarity_new = feats.matmul(Proxies_new_norm)

        # W/ Adapter
        feats_proj_old = self.adapter_old(feats)
        feats_proj_old = F.normalize(feats_proj_old, dim=1)
        # Adapter Reverse
        # Proxies_old_base_norm = self.adapter_old(Proxies_old_base_norm.T).T
        # Proxies_old_base_norm = F.normalize(Proxies_old_base_norm, dim=0)
        # feats_proj_old = F.normalize(feats, dim=1)

        # feats_proj_new = self.adapter_new(feats)
        # feats_proj_new = F.normalize(feats_proj_new, dim=1)
        similarity_old_base = feats_proj_old.matmul(
            Proxies_old_base_norm
        )  # BS * feat_dim X feat_dim * (C_old * num_old_proxy)
        # similarity_local = feats_map.matmul(Proxies_local_norm)
        # similarity_new_base = feats_proj_new.matmul(Proxies_new_base_norm)  # BS * feat_dim X feat_dim * (C_new * num_new_proxy)
        if self.num_proxy_hard:
            similarity_old_hard = feats_proj_old.matmul(
                Proxies_old_hard_norm
            )  # BS * feat_dim X feat_dim * (C_old * num_old_proxy)
            # similarity_new_hard = feats_proj_new.matmul(Proxies_new_hard_norm)  # BS * feat_dim X feat_dim * (C_new * num_new_proxy)

        #     logits_pattern, proj_feats_pattern = self.head_pattern(feats=feats, pattern_idx=0)
        #     logits_pattern_contra = torch.cat([logits_pattern.chunk(2)[0], logits_pattern.chunk(2)[0]])
        # else:
        #     logits_pattern, proj_feats_pattern = self.head_pattern(feats=feats, pattern_idx=0)
        #     logits_pattern_contra = logits_pattern
        # logits_class = torch.zeros([feats.size(0), self.num_classes]).to(feats.device)
        # proj_feats_class = torch.zeros([feats.size(0), self.mlp_out_dim]).to(feats.device)
        # for i in range(self.num_pattern):
        #     idx = (logits_pattern_contra.max(1)[1] == i)
        #     logits_class[idx, :], proj_feats_class[idx, :] = self.head_class(feats=proj_feats_pattern[idx, :], pattern_idx=i)

        out = {
            # "similarity_proxy_old": similarity_proxy_old,
            # "similarity_proxy_new": similarity_proxy_new,
            "similarity_old_base": similarity_old_base,
            # "similarity_local": similarity_local,
            # "similarity_new_base": similarity_new_base,
            # "similarity_old_hard": similarity_old_hard,
            # "similarity_new_hard": similarity_new_hard,
            "feats_proj_old": feats_proj_old,
            # "feats_proj_new": feats_proj_new,
            # "logits_class": logits_class,
            # "proj_feats_class": proj_feats_class,
        }
        if self.num_proxy_hard:
            out["similarity_old_hard"] = similarity_old_hard
        return out

    def forward(self, views):
        if isinstance(views, list):
            embedding = self.encoder(torch.cat(views, dim=0))
            feats = [f for f in embedding[0].chunk(2)]
            # feats_map = [self.norm(self.conv_local(f)) for f in embedding[1].chunk(2)]
            feats_map = [f for f in embedding[1].chunk(2)]
            out = self.forward_proxies(feats, feats_map)
            out_dict = {"feats": feats, "feats_map": feats_map}
            for key in out.keys():
                out_dict[key] = [k for k in out[key].chunk(2)]
            return out_dict
        else:
            embedding = self.encoder(views)
            feats = embedding[0]
            feats_map = embedding[1]
            # feats_map = self.norm(self.conv_local(embedding[1]))
            out = self.forward_proxies(feats, feats_map)
            out["feats"] = feats
            out["feats_map"] = feats_map
            return out


class MultiPatternModel(nn.Module):
    def __init__(
        self,
        model,
        num_old_classes,
        num_new_classes,
        num_pattern=5,
        num_mlp_layers=1,
        hidden_dim=2048,
        mlp_out_dim=256,
        mode="supervised",
    ):
        super().__init__()
        self.encoder = models.__dict__[model](pretrained=True)
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.num_pattern = num_pattern
        self.num_old_classes = num_old_classes
        self.num_new_classes = num_new_classes
        self.mlp_out_dim = mlp_out_dim
        self.hidden_dim = hidden_dim
        self.num_mlp_layers = num_mlp_layers
        self.num_classes = num_old_classes if mode == "supervised" else num_old_classes + num_new_classes
        self.head_class = MultiPatternHead(
            input_dim=mlp_out_dim,
            hidden_dim=hidden_dim,
            output_dim=mlp_out_dim,
            num_prototypes=self.num_classes,
            num_pattern=self.num_pattern,
            num_mlp_layers=num_mlp_layers,
        )
        self.head_pattern = MultiPatternHead(
            input_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            output_dim=mlp_out_dim,
            num_prototypes=num_pattern,
            num_pattern=1,
            num_mlp_layers=num_mlp_layers,
        )
        self.head_pattern_fusion = Prototypes(self.num_pattern * self.num_classes, self.num_classes)

        # self.head_lab = Prototypes(self.feat_dim, num_old_classes)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_pattern.normalize_prototypes()
        self.head_class.normalize_prototypes()

    def forward_heads(self, feats):
        if isinstance(feats, list):
            feats = torch.cat(feats, dim=0)
            logits_pattern, proj_feats_pattern = self.head_pattern(feats=feats, pattern_idx=0)
            logits_pattern_contra = torch.cat([logits_pattern.chunk(2)[0], logits_pattern.chunk(2)[0]])
        else:
            logits_pattern, proj_feats_pattern = self.head_pattern(feats=feats, pattern_idx=0)
            logits_pattern_contra = logits_pattern
        logits_class = torch.zeros([feats.size(0), self.num_classes]).to(feats.device)
        proj_feats_class = torch.zeros([feats.size(0), self.mlp_out_dim]).to(feats.device)
        for i in range(self.num_pattern):
            idx = logits_pattern_contra.max(1)[1] == i
            logits_class[idx, :], proj_feats_class[idx, :] = self.head_class(
                feats=proj_feats_pattern[idx, :], pattern_idx=i
            )
        out = {
            "logits_pattern": logits_pattern,
            "proj_feats_pattern": proj_feats_pattern,
            "logits_class": logits_class,
            "proj_feats_class": proj_feats_class,
            # "logits_"
            # "logits_lab": self.head_lab(F.normalize(feats))
        }
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [f for f in self.encoder(torch.cat(views, dim=0)).chunk(2)]
            out = self.forward_heads(feats)
            out_dict = {"feats": feats}
            for key in out.keys():
                out_dict[key] = [k for k in out[key].chunk(2)]
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out


class MultiHeadResNet(nn.Module):
    def __init__(
        self,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
    ):
        super().__init__()

        # backbone
        self.encoder = models.__dict__[arch](pretrained=True)
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        # modify the encoder for lower resolution
        if low_res:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        self.head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out


class ResNet_(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # backbone
        self.encoder = models.__dict__["resnet50"](pretrained=True)
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.head_lab = Prototypes(self.feat_dim, 99)
        self.Proxies_old_base = nn.Parameter(torch.Tensor(self.feat_dim, 99))
        nn.init.kaiming_uniform_(self.Proxies_old_base, a=math.sqrt(5))

        self.Proxies_new_base = nn.Parameter(torch.Tensor(self.feat_dim, 99))
        nn.init.kaiming_uniform_(self.Proxies_new_base, a=math.sqrt(5))

    def forward_head(self, feats):
        feats_norm = F.normalize(feats)
        out = {"logits_lab": self.head_lab(feats_norm)}
        return out

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()

    def forward(self, views):
        if isinstance(views, list):
            embedding = self.encoder(torch.cat(views, dim=0))
            feats = [f for f in embedding[0].chunk(2)]
            feats_map = [f for f in embedding[1].chunk(2)]
            out = [self.forward_head(f) for f in feats]
            out_dict = {"feats": feats, "feats_map": feats_map, "feats_proj_old": feats}
            for key in out[0].keys():
                out_dict[key] = [o[key] for o in out]
            return out_dict
        else:
            embedding = self.encoder(views)
            feats = embedding[0]
            feats_map = embedding[1]
            out = self.forward_head(feats)
            out["feats"] = feats
            out["feats_proj_old"] = feats
            out["feats_map"] = feats_map
            return out
