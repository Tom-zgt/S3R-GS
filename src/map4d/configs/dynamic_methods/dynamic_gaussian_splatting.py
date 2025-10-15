"""Dynamic Gaussian Splatting method."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from map4d.data.datamanagers.dynamic_datamanager import DynamicDataManagerConfig
from map4d.model.gauss_splat import NeuralDynamicGaussianSplattingConfig
from map4d.pipelines.dynamic_pipeline import DynamicPipelineConfig


dynamic_gaussian_splatting_method = MethodSpecification(
    config=TrainerConfig(
        method_name="dynamic-gaussian-splatting",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicPipelineConfig(
            datamanager=DynamicDataManagerConfig(
                _target=DynamicDataManagerConfig,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=VanillaDataManagerConfig.camera_optimizer,
            ),
            model=NeuralDynamicGaussianSplattingConfig(
                _target=NeuralDynamicGaussianSplattingConfig,
                calc_fid_steps=(20000,),  # Calculate FID at step 20000
                ray_patch_size=(32, 32),
            ),
        ),
        optimizers={
            "gaussian": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "camera_opt": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
        },
        schedulers={
            "gaussian": ExponentialDecaySchedulerConfig(
                lr_final=0.0001,
                max_steps=30000,
            ),
            "camera_opt": ExponentialDecaySchedulerConfig(
                lr_final=1e-4,
                max_steps=30000,
            ),
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Dynamic Gaussian Splatting method.",
) 