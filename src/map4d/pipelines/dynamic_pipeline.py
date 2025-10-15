from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image, to_tensor

from map4d.model.gauss_splat import NeuralDynamicGaussianSplattingModel, NeuralDynamicGaussianSplattingConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.utils import profiler


@dataclass
class DynamicPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DynamicPipeline)
    """target class to instantiate"""
    model: NeuralDynamicGaussianSplattingConfig = field(default_factory=NeuralDynamicGaussianSplattingConfig)
    """specifies the model config"""
    calc_fid_steps: Tuple[int, ...] = (20000,)  # NOTE: must also be an eval step for this to work
    """Whether to calculate FID for lane shifted images."""
    ray_patch_size: Tuple[int, int] = (32, 32)
    """Size of the ray patches to sample from the image during training (for camera rays only)."""

    def __post_init__(self) -> None:
        assert self.ray_patch_size[0] == self.ray_patch_size[1], "Non-square patches are not supported yet, sorry."
        self.datamanager.image_divisible_by = self.model.rgb_upsample_factor


class DynamicPipeline(VanillaPipeline):
    """Pipeline for training dynamic models."""

    def __init__(self, config: DynamicPipelineConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Fix type hints
        self.model: NeuralDynamicGaussianSplattingModel = self.model
        self.config: DynamicPipelineConfig = self.config

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            lane_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (0, 2, 3)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            vertical_shift_fids = (
                {i: FrechetInceptionDistance().to(self.device) for i in (1,)}
                if step in self.config.calc_fid_steps or step is None
                else {}
            )

            num_images = len(self.datamanager.fixed_indices_eval_dataloader)
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # Generate images from the original rays
                camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                # Compute metrics for the original image
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                        )
                metrics_dict_list.append(metrics_dict)
                if lane_shift_fids:
                    self.model._update_lane_shift_fid(lane_shift_fids, camera_ray_bundle, batch["image"], outputs["rgb"])
                if vertical_shift_fids:
                    self.model._update_vertical_shift_fid(vertical_shift_fids, camera_ray_bundle, batch["image"])
                progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        keys = {key for metrics_dict in metrics_dict_list for key in metrics_dict.keys()}

        for key in keys:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                    )
                )

        # Add FID metrics (if applicable)
        for shift, fid in lane_shift_fids.items():
            metrics_dict[f"lane_shift_{shift}_fid"] = fid.compute().item()

        for shift, fid in vertical_shift_fids.items():
            metrics_dict[f"vertical_shift_{shift}_fid"] = fid.compute().item()

        self.train()
        return metrics_dict 