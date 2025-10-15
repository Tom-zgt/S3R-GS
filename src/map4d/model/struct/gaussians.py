from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.nn import Parameter
import wandb
# wandb.init()

# class Gaussians(torch.nn.Module):
#     """Data structure for 3D Gaussian representation."""

#     def __init__(
#         self,
#         means: Tensor,
#         opacities: Tensor,
#         scales: Tensor,
#         quats: Tensor,
#         other_attrs: dict | None = None,
#     ) -> None:
#         super().__init__()
#         self.means = Parameter(means)

#         assert len(opacities.shape) == 2, "Expected shape N, 1"
#         self._opacities = Parameter(opacities)
#         assert len(self.means) == len(self._opacities)

#         self._scales = Parameter(scales)
#         assert len(self.means) == len(self._scales)

#         self._quats = Parameter(quats)
#         assert len(self.means) == len(self._quats)

#         self._other_keys = []
#         if other_attrs is not None:
#             self._other_keys = list(other_attrs.keys())
#             for key, value in other_attrs.items():
#                 assert len(value.shape) == 2, "Expected shape N, C"
#                 setattr(self, key, Parameter(value, requires_grad=value.requires_grad))
#                 assert len(self.means) == len(value)

#     @property
#     def num_points(self):
#         return self.means.shape[0]

#     @property
#     def quats(self):
#         if hasattr(self, "_quats"):
#             return self._quats
#         return None

#     @property
#     def scales(self):
#         if hasattr(self, "_scales"):
#             return self._scales
#         return None

#     @property
#     def opacities(self):
#         if hasattr(self, "_opacities"):
#             return self._opacities
#         return None

#     @property
#     def others(self):
#         if hasattr(self, "_other_keys"):
#             return {key: getattr(self, key) for key in self._other_keys}
#         return None

#     @property
#     def device(self):
#         return self.means.device

#     def get_param_groups(self) -> Dict[str, Parameter]:
#         """Get Gaussian parameters for optimization."""
#         params = {"xyz": self.means}
#         params["opacity"] = self.opacities
#         params["scales"] = self.scales
#         params["quats"] = self.quats
#         for k in self._other_keys:
#             v = getattr(self, k)
#             if v.requires_grad:
#                 params[k] = v
#         return params

#     def add_attribute(self, key: str, value: Tensor, requires_grad: bool = True):
#         """Add an attribute to the Gaussians object."""
#         assert len(self.means) == len(value)
#         setattr(self, key, Parameter(value, requires_grad=requires_grad))
#         if not hasattr(self, "_other_keys"):
#             self._other_keys = []
#         self._other_keys.append(key)

#     def cat(self, other: Gaussians):
#         means = torch.cat([self.means.data, other.means.data], dim=0)
#         opacities = torch.cat([self.opacities.data, other.opacities.data], dim=0)
#         scales = torch.cat([self.scales.data, other.scales.data], dim=0)
#         quats = torch.cat([self.quats.data, other.quats.data], dim=0)
#         others = (
#             {key: torch.cat([getattr(self, key).data, getattr(other, key).data], dim=0) for key in self._other_keys}
#             if hasattr(self, "_other_keys")
#             else None
#         )
#         return Gaussians(
#             means,
#             opacities=opacities,
#             scales=scales,
#             quats=quats,
#             other_attrs=others,
#         )

#     def detach(self):
#         means = self.means.data.detach()
#         opacities = self.opacities.data.detach()
#         scales = self.scales.data.detach()
#         quats = self.quats.data.detach()
#         others = (
#             {key: getattr(self, key).data.detach() for key in self._other_keys}
#             if hasattr(self, "_other_keys")
#             else None
#         )
#         return Gaussians(
#             means,
#             opacities=opacities,
#             scales=scales,
#             quats=quats,
#             other_attrs=others,
#         )

#     @classmethod
#     def empty_like(cls, gaussians: Gaussians) -> Gaussians:
#         """Create an empty instance of Gaussians with the same feature shapes and attributes as the input."""
#         means = torch.empty((0, 3), dtype=gaussians.means.dtype, device=gaussians.means.device)
#         opacities = torch.empty((0, 1), dtype=gaussians.opacities.dtype, device=gaussians.opacities.device)
#         scales = torch.empty((0, 3), dtype=gaussians.scales.dtype, device=gaussians.scales.device)
#         quats = torch.empty((0, 4), dtype=gaussians.quats.dtype, device=gaussians.quats.device)
#         others = (
#             {
#                 key: torch.empty(
#                     (0, getattr(gaussians, key).shape[1]) if len(getattr(gaussians, key).shape) > 1 else (0,),
#                     dtype=getattr(gaussians, key).dtype,
#                     device=getattr(gaussians, key).device,
#                 )
#                 for key in gaussians._other_keys
#             }
#             if hasattr(gaussians, "_other_keys")
#             else None
#         )
#         return Gaussians(
#             means,
#             opacities=opacities,
#             scales=scales,
#             quats=quats,
#             other_attrs=others,
#         )

#     def __getitem__(self, idx):
#         """Gaussian slicing. NOTE: This function breaks gradient flow!"""
#         means = self.means.data[idx]
#         opacities = self.opacities.data[idx]
#         scales = self.scales.data[idx]
#         quats = self.quats.data[idx]
#         others = (
#             {key: getattr(self, key).data[idx] for key in self._other_keys} if hasattr(self, "_other_keys") else None
#         )
#         return Gaussians(
#             means,
#             opacities=opacities,
#             scales=scales,
#             quats=quats,
#             other_attrs=others,
#         )

#     def to_dict(self):
#         """Returns a dictionary representation of the Gaussians object."""
#         attributes = {
#             "means": self.means,
#             "opacities": self.opacities,
#             "scales": self.scales,
#             "quats": self.quats,
#         }
#         attributes.update({key: getattr(self, key) for key in self._other_keys} if hasattr(self, "_other_keys") else {})
#         return attributes

#     def __repr__(self) -> str:
#         """Returns a dictionary like visualization of all attributes of the Gaussians object.

#         Iterates through all attributes of the Gaussians object and returns a string representation of the attributes.
#         """
#         attributes = self.to_dict()
#         return (
#             "Gaussians(" + ", ".join([f"{k}={v.data if v is not None else None}" for k, v in attributes.items()]) + ")"
#         )

class Gaussians(torch.nn.Module):
    """Data structure for 3D Gaussian representation."""

    def __init__(
        self,
        means: Tensor,
        opacities: Tensor,
        scales: Tensor,
        quats: Tensor,
        scene_timestamps: Tensor | None = None,
        scene_seqs: Tensor | None = None,
        scene_views: Tensor | None = None,
        views: int = 1,
        seqs: int = 1,
        record_train_frames: Tensor | None = None,
        other_attrs: dict | None = None,
    ) -> None:
        super().__init__()
        self.frames = 0 if scene_timestamps is None else scene_timestamps.shape[0] // (seqs*views)
        self.views = views
        self.seqs = seqs
        self.scene_timestamps = scene_timestamps if scene_timestamps is None else scene_timestamps.to(means.device)
        self.scene_seqs = scene_seqs if scene_seqs is None else scene_seqs.to(means.device)
        self.scene_views = scene_views if scene_views is None else scene_views.to(means.device)
        # print(self.scene_timestamps.shape[0])
        self.record_train_frames = None if scene_timestamps is None else record_train_frames #torch.zeros((self.scene_timestamps.shape[0]),dtype=torch.bool,device=means.device)
        self.means = Parameter(means)

        assert len(opacities.shape) == 2, "Expected shape N, 1"
        self._opacities = Parameter(opacities)
        assert len(self.means) == len(self._opacities)

        self._scales = Parameter(scales)
        assert len(self.means) == len(self._scales)

        self._quats = Parameter(quats)
        assert len(self.means) == len(self._quats)

        # self._temporal_vis = torch.zeros((self.means.shape[0],frames*views),dtype=torch.bool,device=self.means.device)
        # self._point_life = torch.zeros((self.means.shape[0],frames*views))
        self._other_keys = []
        if other_attrs is not None:
            self._other_keys = list(other_attrs.keys())
            for key, value in other_attrs.items():
                assert len(value.shape) == 2, "Expected shape N, C"
                setattr(self, key, Parameter(value, requires_grad=value.requires_grad))
                assert len(self.means) == len(value)

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def quats(self):
        if hasattr(self, "_quats"):
            return self._quats
        return None

    @property
    def scales(self):
        if hasattr(self, "_scales"):
            return self._scales
        return None

    @property
    def opacities(self):
        if hasattr(self, "_opacities"):
            return self._opacities
        return None

    @property
    def others(self):
        if hasattr(self, "_other_keys"):
            return {key: getattr(self, key) for key in self._other_keys}
        return None

    @property
    def device(self):
        return self.means.device

    def get_param_groups(self) -> Dict[str, Parameter]:
        """Get Gaussian parameters for optimization."""
        params = {"xyz": self.means}
        params["opacity"] = self.opacities
        params["scales"] = self.scales
        params["quats"] = self.quats
        for k in self._other_keys:
            v = getattr(self, k)
            if v.requires_grad:
                params[k] = v
        return params

    def add_attribute(self, key: str, value: Tensor, requires_grad: bool = True):
        """Add an attribute to the Gaussians object."""
        assert len(self.means) == len(value)
        setattr(self, key, Parameter(value, requires_grad=requires_grad))
        if not hasattr(self, "_other_keys"):
            self._other_keys = []
        self._other_keys.append(key)

    def cat(self, other: Gaussians):
        means = torch.cat([self.means.data, other.means.data], dim=0)
        opacities = torch.cat([self.opacities.data, other.opacities.data], dim=0)
        scales = torch.cat([self.scales.data, other.scales.data], dim=0)
        quats = torch.cat([self.quats.data, other.quats.data], dim=0)
        others = (
            {key: torch.cat([getattr(self, key).data, getattr(other, key).data], dim=0) for key in self._other_keys}
            if hasattr(self, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
            scene_timestamps=self.scene_timestamps,
            scene_seqs=self.scene_seqs,
            scene_views=self.scene_views,
            views=self.views,
            seqs=self.seqs,
            record_train_frames=self.record_train_frames,
        )

    def detach(self):
        means = self.means.data.detach()
        opacities = self.opacities.data.detach()
        scales = self.scales.data.detach()
        quats = self.quats.data.detach()
        others = (
            {key: getattr(self, key).data.detach() for key in self._other_keys}
            if hasattr(self, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
            scene_timestamps=self.scene_timestamps,
            scene_seqs=self.scene_seqs,
            scene_views=self.scene_views,
            views=self.views,
            seqs=self.seqs,
            record_train_frames=self.record_train_frames,
        )

    @classmethod
    def empty_like(cls, gaussians: Gaussians) -> Gaussians:
        """Create an empty instance of Gaussians with the same feature shapes and attributes as the input."""
        means = torch.empty((0, 3), dtype=gaussians.means.dtype, device=gaussians.means.device)
        opacities = torch.empty((0, 1), dtype=gaussians.opacities.dtype, device=gaussians.opacities.device)
        scales = torch.empty((0, 3), dtype=gaussians.scales.dtype, device=gaussians.scales.device)
        quats = torch.empty((0, 4), dtype=gaussians.quats.dtype, device=gaussians.quats.device)
        others = (
            {
                key: torch.empty(
                    (0, getattr(gaussians, key).shape[1]) if len(getattr(gaussians, key).shape) > 1 else (0,),
                    dtype=getattr(gaussians, key).dtype,
                    device=getattr(gaussians, key).device,
                )
                for key in gaussians._other_keys
            }
            if hasattr(gaussians, "_other_keys")
            else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
            scene_timestamps=gaussians.scene_timestamps,
            scene_seqs=gaussians.scene_seqs,
            scene_views=gaussians.scene_views,
            views=gaussians.views,
            seqs=gaussians.seqs,
            record_train_frames=gaussians.record_train_frames,
        )
        
    def get_mask(self, t, view, seq_id, train=False):
        # print(t,view,seq_id)
        if train:
            # _, time_idcs = torch.topk((self.scene_timestamps-t).abs(), 1, dim=0, largest=False)
            time_errors = (self.scene_timestamps - t).abs()

            # 仅筛选符合 `seq` 和 `view` 的索引
            valid_mask = (self.scene_seqs == seq_id) & (self.scene_views == view)

            # 仅保留符合 `seq` 和 `view` 的时间误差
            filtered_time_errors = torch.where(valid_mask, time_errors, torch.tensor(float('inf')))

            # 找到最接近 `t` 的索引
            _, time_idcs = torch.min(filtered_time_errors, dim=0)
            time_idcs = time_idcs[0]
            # time_idcs = torch.where(torch.isclose(t, self.scene_timestamps, atol=1e-8))[0]
            # print(view)
            # print(time_idcs)
            # time_idcs = time_idcs[view]
            # time_idcs = time_idcs[0][0]
            mask = (self.temporal_vis[:,0+2*(view)] <= time_idcs) & (self.temporal_vis[:,1+2*(view)] > time_idcs)
            # if mask.sum() == 0:
            #     print(time_idcs, self.temporal_vis[:,0], self.temporal_vis[:,1])
            # print(mask.shape)
            return mask[:,0]
        else:
            return torch.ones_like(self.temporal_vis[:,0])
        
    def update_vis(self, t,view,seq_id, mask):
        # _, time_idcs = torch.topk((self.scene_timestamps-t).abs(), 1, dim=0, largest=False)
        # time_idcs = time_idcs[0][0]
        time_errors = (self.scene_timestamps - t).abs()

        # 仅筛选符合 `seq` 和 `view` 的索引
        valid_mask = (self.scene_seqs == seq_id) & (self.scene_views == view)

        # 仅保留符合 `seq` 和 `view` 的时间误差
        filtered_time_errors = torch.where(valid_mask, time_errors, torch.tensor(float('inf')))

        # 找到最接近 `t` 的索引
        _, time_idcs = torch.min(filtered_time_errors, dim=0)
        time_idcs = time_idcs[0]
        self.record_train_frames[time_idcs] = True
        self.point_life[mask,2*(view)] = torch.min(torch.ones_like(self.point_life[mask,2*(view)])*time_idcs, self.point_life[mask,2*(view)]) 
        self.point_life[mask,1+2*(view)] = torch.max(torch.ones_like(self.point_life[mask,1+2*(view)])*(1+time_idcs), self.point_life[mask,1+2*(view)]) 
        # print(self.record_train_frames.sum().float()/self.record_train_frames.shape[0])
        # wandb.log({"record ratio": (self.record_train_frames.sum().float()/self.record_train_frames.shape[0]).item()})
        if (self.record_train_frames.sum().float()/self.record_train_frames.shape[0]) == 1.0:
            print("reset_epoch")
            self.reset_epoch()
        # time_idcs = torch.sort(time_idcs)[0]
        # self.point_life[mask,time_idcs[1][0]:time_idcs[0][0]+1] = 0
        # self.point_life[(~mask) & self.temporal_vis[:,time_idcs[1][0]],time_idcs[1][0]] = self.point_life[~mask,time_idcs[1][0]] - 1
        # self.point_life[(~mask) & self.temporal_vis[:,time_idcs[0][0]],time_idcs[0][0]] = self.point_life[~mask,time_idcs[0][0]] - 1
        # self.temporal_vis[:,time_idcs[1][0]:time_idcs[0][0]+1] = self.point_life[:,time_idcs[1][0]:time_idcs[0][0]+1] > -3

    def reset_epoch(self):
        self.temporal_vis[..., ::2] = (self.point_life[..., ::2] - 1)  # （0, 2, 4...）
        self.temporal_vis[..., 1::2] = (self.point_life[..., 1::2] + 1) 
        other_mask = (self.point_life[..., ::2] == self.scene_timestamps.shape[0]) & (self.point_life[:,1::2] == 0)
        self.temporal_vis[...,::2][other_mask] = 0
        self.temporal_vis[:,1::2][other_mask] = self.scene_timestamps.shape[0]
        # print(self.temporal_vis[:,0])
        # print(self.temporal_vis[:,1])
        self.point_life[..., ::2] = self.scene_timestamps.shape[0]
        self.point_life[:,1::2] = 0
        self.record_train_frames[:] = False
        
    def reset_tv(self):
        self.temporal_vis[:,::2] = 0
        self.temporal_vis[:,1::2] = self.scene_timestamps.shape[0]
        self.record_train_frames[:] = False
    
    def __getitem__(self, idx):
        """Gaussian slicing. NOTE: This function breaks gradient flow!"""
        means = self.means.data[idx]
        opacities = self.opacities.data[idx]
        scales = self.scales.data[idx]
        quats = self.quats.data[idx]
        others = (
            {key: getattr(self, key).data[idx] for key in self._other_keys} if hasattr(self, "_other_keys") else None
        )
        return Gaussians(
            means,
            opacities=opacities,
            scales=scales,
            quats=quats,
            other_attrs=others,
            scene_timestamps=self.scene_timestamps,
            scene_seqs=self.scene_seqs,
            scene_views=self.scene_views,
            views=self.views,
            seqs=self.seqs,
            record_train_frames=self.record_train_frames,
        )

    def to_dict(self):
        """Returns a dictionary representation of the Gaussians object."""
        attributes = {
            "means": self.means,
            "opacities": self.opacities,
            "scales": self.scales,
            "quats": self.quats,
        }
        attributes.update({key: getattr(self, key) for key in self._other_keys} if hasattr(self, "_other_keys") else {})
        return attributes

    def __repr__(self) -> str:
        """Returns a dictionary like visualization of all attributes of the Gaussians object.

        Iterates through all attributes of the Gaussians object and returns a string representation of the attributes.
        """
        attributes = self.to_dict()
        return (
            "Gaussians(" + ", ".join([f"{k}={v.data if v is not None else None}" for k, v in attributes.items()]) + ")"
        )
