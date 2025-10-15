"""Dynamic 3D Gaussian Fields Model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import torchdiffeq  
import torch
import torch.distributed
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from lpips import lpips
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SE3
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color

# need following import for background color override
from nerfstudio.utils.comms import get_rank
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.viewer import Viewer, ViewerControl
from nerfstudio.viewer.viewer_elements import ViewerSlider
from pytorch_msssim import SSIM
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
import copy
from map4d.cameras.camera_optimizer import VehiclePoseOptimizerConfig
from map4d.common.geometry import quaternion_multiply
from map4d.common.metric import psnr
from map4d.common.metric import ssim as ssim_fn
from map4d.common.visualize import (
    apply_depth_colormap,
    draw_boxes_bev,
    draw_boxes_in_image,
    draw_points_bev,
    get_canvas_bev,
    rasterize_points,
)
from map4d.cuda import VideoEmbedding
from map4d.model.controller.density_control import VanillaAdaptiveDensityController
from map4d.model.gaussian_field.dynamic import DynamicFieldHead
from map4d.model.gaussian_field.static import StaticFieldHead
from map4d.model.gaussian_field.util import initialize_gaussians
from map4d.model.util import (
    c2w_to_local,
    c2w_to_local_grad,
    get_objects_at_time,
    mask_images,
    opengl_frustum_check,
    frustum_check,
    resize_image,
    trigger_reducer_update,
)
import time
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from pathlib import Path
import wandb
# wandb.init()
# this controls the tile size of rasterization, 16 is a good default
BLOCK_WIDTH = 16


def _patch_broken_nerfstudio_viewer():
    from nerfstudio.viewer_legacy.server.viewer_utils import SetTrace

    # nerfstudio's viewer has two rendering passes for each view - fast and slow.
    # First, the fast pass is used with a lower res to render something to present to the user immediately.
    # This is followed by the second pass which takes much longer for nerfs
    # (could be seconds with nerfacto). For that reason, nerfstudio uses sys.settrace
    # to cancel the running rendering and switch back to faster rendering as soon as
    # user moves the camera. Unfortunately, this has lots of unwanted effects - breaks
    # debuggers, and in our case, it cancels the rendering at a point that breaks the model,
    # causes random errors, or renders a bad image.
    # This function patches NerfStudio such that it does not set the trace and break the viewer.
    # Patch broken sys.settrace in nerfstudio
    SetTrace.__enter__ = lambda self: self


_patch_broken_nerfstudio_viewer()


@dataclass
class NeuralDynamicGaussianSplattingConfig(ModelConfig):
    """Dynamic 3D Gaussian Fields model configuration.

    Args:
        DynamicGaussianSplattingModelConfig: Dynamic Gaussian Splatting model configuration
    """

    _target: Type = field(default_factory=lambda: NeuralDynamicGaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    reset_tv_every: int = 500
    """Every this many refinement steps, reset the temporal visibilites"""
    densify_grad_thresh: float = 0.0006
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    near_clip_thresh: float = 1.0
    """near clip threshold for projection of gaussians in meters."""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """camera optimizer config"""
    optimize_box_poses: bool = False
    """If enabled, the box poses are optimized as well."""

    # primitive growth control
    max_num_gaussians: int = 4750000
    """Maximum number of gaussians (not enforced exactly, but softly)."""

    # static field settings
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""

    # dynamic field settings
    object_grid_num_levels: int = 8
    """Number of levels of the hashmap for the objects."""
    object_grid_max_res: int = 1024
    """Maximum resolution of the hashmap for the objects."""
    object_grid_log2_hashmap_size: int = 15
    """Size of the hashmap for the objects."""
    object_mlp_num_layers: int = 2
    """Number of layers in the object MLP(s)."""
    object_mlp_layer_width: int = 64
    """Width of the layers in the object MLP(s)."""
    encode_time: bool = True
    """Whether to encode time in the object color head."""
    model_deformation: bool = True
    """Whether to model deformation in the object field head."""

    # sensor codes
    sensor_appearance_embedding_dim: int = 0
    """Dimension of the sensor appearance embeddings."""

    # scene / object codes
    appearance_embedding_dim: int = 32
    """Dimension of the scene appearance embeddings."""
    transient_embedding_dim: int = 32
    """Dimension of the scene transient embeddings."""
    video_embedding_frequencies: int = 6
    """Frequencies for video embeddings (scene appearance / transient)."""

    # depth visualization settings
    min_depth: float = 1.0
    """Minimum depth for visualization."""
    max_depth: float = 82.5
    """Maximum depth for visualization."""
    depth_loss_mult: float = 0.0
    """Lambda of the depth loss."""

    # metric computation for waymo
    is_dynamic_mask: bool = False
    """If True, the mask from the dataloader splits dynamic and static content and to mask out e.g. ego-vehicle."""

    # other settings
    optimize_object_height: bool = True
    """Optimize the object height."""
    adc_min_num_visible: int | float = 0
    """Trigger adaptive density control only for gaussians with sufficient observations."""
    is_static: bool = False
    """If True, the model is static, i.e. no dynamic Gaussians or fields."""
    cull_within_scene_bounds_only: bool = True
    """If True, cull huge gaussians only within the scene bounds."""
    instance_proj: bool = True
    """If True, proj object gaussians with instance-specific projection."""
    temporal_vis: bool = False
    """If True, initial scene gaussians with temporal visibilities."""
    temporal_view: int = 1
    temporal_seq: int = 1
    autolod: bool = False
    lod: int = 6
    nus: bool = False
    max_dis: float = 80.0

    # FID test settings
    calc_fid_steps: Tuple[int, ...] = (40000,)  # NOTE: must also be an eval step for this to work
    """Whether to calculate FID for lane shifted images."""
    # ray_patch_size: Tuple[int, int] = (32, 32)
    # """Size of the ray patches to sample from the image during training (for camera rays only)."""

def P(d):
    return 0.95 - (0.95 - 0.01) * torch.max(0.5-d,torch.zeros_like(d))
def P_real(d, max_dis=80.0):
    return 0.95 - (0.95 - 0.01) * torch.max((max_dis-d)/max_dis,torch.zeros_like(d))

class NeuralDynamicGaussianSplattingModel(Model):
    """Dynamic 3D Gaussian Fields model.

    Args:
        config: configuration to instantiate model
    """

    config: NeuralDynamicGaussianSplattingConfig

    def __init__(self, *args, num_eval_data: int, test_mode: str = "val", **kwargs):
        train_meta = kwargs.get("train_metadata", None)
        eval_meta = kwargs.get("eval_metadata", None)
        self.num_cameras = train_meta["num_cameras"]
        self.num_eval_data = num_eval_data
        self.test_mode = test_mode
        self.dataparser_scale = train_meta["dataparser_scale"]
        self.train_camera_seq_ids = train_meta["sequence_ids"]
        self.train_camera_times = train_meta["frame_ids"]
        self.train_cameras = kwargs.get("train_cameras", None)

        self.eval_camera_seq_ids = eval_meta["sequence_ids"]
        self.eval_camera_times = eval_meta["frame_ids"]
        self.eval_cameras = kwargs.get("eval_cameras", None)

        # scene info
        self.sequence_ids = train_meta["sequence_ids"].unique()
        assert (self.sequence_ids == torch.arange(self.sequence_ids.numel())).all()

        self.object_seq_ids = train_meta["object_seq_ids"]
        self.object_class_ids = train_meta["object_class_ids"]
        self.object_times = train_meta["object_times"]
        self.object_poses = train_meta["object_poses"]
        self.object_dims = train_meta["object_dims"]
        self.object_ids = train_meta["object_ids"]
        self.intrain = False
        self.num_objects = (
            torch.cat(self.object_ids).unique().numel()
            if isinstance(self.object_ids, list)
            else self.object_ids.unique().numel()
        )
        print("object_number:",self.num_objects)
        assert (torch.cat(self.object_ids).unique() == torch.arange(self.num_objects)).all()
        
        
        # init points
        if "seed_points" in train_meta:
            self.seed_pts = (train_meta["seed_points"], train_meta["seed_colors"])
            self.object_pts = (train_meta["object_points"], train_meta["object_colors"])
        else:
            # mockup seed points
            self.seed_pts = torch.rand((1000, 3), device="cuda"), torch.rand((1000, 3), device="cuda") * 255
            self.object_pts = (
                [torch.zeros((1000, 3), device="cuda") for _ in range(self.num_objects)],
                [torch.rand((1000, 3), device="cuda") * 255 for _ in range(self.num_objects)],
            )
            CONSOLE.log("Created Mockup seed points. Make sure to load from checkpoint.")

        # transform lists to tensors while preserving dimensions, pad entries
        if isinstance(self.object_ids, list):

            def _to_tensor(x, pad_val=0):
                max_shape = torch.tensor([y.shape for y in x]).max(dim=0)[0]
                x_tensor = torch.ones((len(x), *max_shape)) * pad_val
                for i, y in enumerate(x):
                    x_tensor[i, : len(y)] = y
                return x_tensor

            self.object_ids = _to_tensor(self.object_ids, pad_val=-1)
            self.object_poses = _to_tensor(self.object_poses)
            self.object_dims = _to_tensor(self.object_dims)
            self.object_class_ids = _to_tensor(self.object_class_ids)
        self.num_classes = self.object_class_ids.unique().numel()
        
        super().__init__(*args, **kwargs)

        # create parameters for objects
        self.object_poses = Parameter(self.object_poses, requires_grad=False)
        self.object_poses_delta = Parameter(
            torch.zeros_like(self.object_poses), requires_grad=self.config.optimize_box_poses
        )
        self.object_ids_init = self.object_ids.clone()
        self.object_dims = Parameter(self.object_dims, requires_grad=False)
        self.object_ids = Parameter(self.object_ids, requires_grad=False)
        self.object_seq_ids = Parameter(self.object_seq_ids, requires_grad=False)
        self.object_class_ids = Parameter(self.object_class_ids, requires_grad=False)
        self.object_times = Parameter(self.object_times, requires_grad=False)
        self.render_images = False  # Flag used to render debugging images
        
        # add viewer controls
        self.viewer_time = ViewerSlider("Time", 0.0, -1.0, 1.0, 0.01)
        if self.sequence_ids.numel() > 1:
            self.viewer_sequence = ViewerSlider("Sequence", 0, 0, self.sequence_ids.numel() - 1, 1.0)
        else:
            self.viewer_sequence = None

        # Callback to register viewer elements
        class _ViewerControlHook(ViewerControl):
            def __init__(self, *args, **kwargs):
                pass

            def _setup(_self, viewer: Viewer):
                self.register_viewer_elements(viewer)

        self.viewer_control_hook = _ViewerControlHook()

    def register_viewer_elements(self, viewer: Viewer):
        # The legacy component api is awful, so we have to do this
        def add_callback(component, cb):
            component.gui_handle.on_update(cb)
            if component.gui_handle is None:
                old_hook = component.cb_hook

                def hook(*args, **kwargs):
                    cb()
                    if old_hook is not None:
                        old_hook(*args, **kwargs)

                self.viewer_time.cb_hook = hook
            else:
                component.gui_handle.on_update(cb)

        add_callback(self.viewer_time, lambda *args, **kwargs: viewer._trigger_rerender())
        if self.viewer_sequence is not None:
            add_callback(self.viewer_sequence, lambda *args, **kwargs: viewer._trigger_rerender())

    def populate_modules(self):
        # init points
        assert self.seed_pts is not None
        scene_pts, scene_cols = self.seed_pts
        assert self.object_pts is not None
        obj_pts, obj_cols = self.object_pts
        # min init points for objects
        device = obj_pts[0].device
        for i, dim in enumerate(self._get_dim_per_obj()):
            num_pts = obj_pts[i].shape[0]
            if num_pts < 32:
                dim = dim.to(device)
                obj_pts[i] = torch.cat([obj_pts[i], (torch.rand((32 - num_pts, 3), device=device) - 0.5) * dim], dim=0)
                obj_cols[i] = torch.cat([obj_cols[i], torch.rand((32 - num_pts, 3), device=device) * 255], dim=0)
        self.object_pts = (obj_pts, obj_cols)
        num_object_pts = sum([pts.shape[0] for pts in obj_pts])
        # limit num init points
        if self.config.max_num_gaussians > 0 and scene_pts.shape[0] > self.config.max_num_gaussians:
            scene_num = self.config.max_num_gaussians - num_object_pts
            CONSOLE.log(
                f"Scene points plus object points exceeds max num gaussians {self.config.max_num_gaussians}: {scene_pts.shape[0]}, {num_object_pts}"
            )
            CONSOLE.log(f"Subsampling {scene_pts.shape[0]} scene points to {scene_num}")
            indices = torch.randperm(scene_pts.shape[0], device=scene_pts.device)[:scene_num]
            scene_pts, scene_cols = scene_pts[indices], scene_cols[indices]
            self.seed_pts = (scene_pts, scene_cols)

        # Node latent codes
        # scene level codes, per-gaussian features, fields (opacities, color)
        num_seq = self.sequence_ids.numel()
        if num_seq > 1 and self.config.appearance_embedding_dim > 0:
            assert self.config.sensor_appearance_embedding_dim == 0
            # Dimension of fourier encoding applied to time when computing appearance embeddings
            self.scene_embedding_appearance = VideoEmbedding(
                num_seq, self.config.video_embedding_frequencies, self.config.appearance_embedding_dim
            )
        else:
            self.config.appearance_embedding_dim = 0

        if self.config.sensor_appearance_embedding_dim > 0:
            self.sensor_embedding_appearance = VideoEmbedding(
                self.num_cameras, self.config.video_embedding_frequencies, self.config.sensor_appearance_embedding_dim
            )
            self.config.appearance_embedding_dim = self.config.sensor_appearance_embedding_dim

        if num_seq > 1 and self.config.transient_embedding_dim > 0:
            self.scene_embedding_transient = VideoEmbedding(
                num_seq, self.config.video_embedding_frequencies, self.config.transient_embedding_dim
            )
        else:
            self.config.transient_embedding_dim = 0

        # scene points init
        assert self.seed_pts is not None and not self.config.random_init

        self.scene_field = StaticFieldHead(
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            transient_embedding_dim=self.config.transient_embedding_dim,
            spatial_distortion=SceneContraction(order=float("inf")),
        )
        if self.config.temporal_vis:
            self.scene_gaussians = initialize_gaussians(self.seed_pts[0],w_temporal_vis=True,scene_timestamps=self.train_cameras.times,scene_seqs=self.train_cameras.metadata['sequence_ids'],scene_views=self.train_cameras.metadata['camera_ids'],view_nums=1 if self.config.temporal_view == -1 else self.train_cameras.metadata['camera_ids'].unique().shape[0], seq_nums=1 if self.config.temporal_seq == -1 else self.train_cameras.metadata['sequence_ids'].unique().shape[0])
        else:
            self.scene_gaussians = initialize_gaussians(self.seed_pts[0])

        field_head_kwargs = dict(
            feature_dim=16,  # output dimension of feature hashgrid
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            time_dependent_appearance=self.config.encode_time,
            num_levels=self.config.object_grid_num_levels,
            max_res=self.config.object_grid_max_res,
            log2_hashmap_size=self.config.object_grid_log2_hashmap_size,
            head_mlp_num_layers=self.config.object_mlp_num_layers,
            head_mlp_layer_width=self.config.object_mlp_layer_width,
        )

        self.dynamic_fields = torch.nn.ModuleList()
        if not self.config.is_static:
            for i in range(self.num_classes):
                model_deform = i == 1 and self.config.model_deformation
                self.dynamic_fields.append(DynamicFieldHead(**field_head_kwargs, time_dependent_geometry=model_deform))

            total_obj_gaussians = None
            total_obj_ids = []
            for obj_id, (pts, obj_dim) in enumerate(zip(self.object_pts[0], self._get_dim_per_obj())):
                assert (obj_dim > 1e-6).all(), "Dimensions must be positive."
                pts = pts / obj_dim.to(pts.device).max(dim=-1)[0].unsqueeze(-1)
                assert (pts.abs() <= 0.5).all(), f"Expected normalized coordinates, got {pts.max(), pts.min()}."
                if total_obj_gaussians is None:
                    total_obj_gaussians = initialize_gaussians(pts)
                else:
                    total_obj_gaussians = total_obj_gaussians.cat(initialize_gaussians(pts))
                total_obj_ids.append(torch.full((pts.shape[0],), obj_id, device=pts.device, dtype=torch.int32))
            self.object_gaussians = total_obj_gaussians
            self.object_gaussians.add_attribute(
                "object_ids", torch.cat(total_obj_ids).unsqueeze(-1), requires_grad=False
            )

        # delete seed_pts and object_pts
        del self.seed_pts
        del self.object_pts

        # adaptive density control
        self.adaptive_density_control = []
        for i, gaussians in enumerate(self._get_gaussians()):
            if i == 0 and self.config.cull_within_scene_bounds_only:
                scene_box = self.scene_box
            else:
                scene_box = None
            if self.config.temporal_vis and (i == 0):
                self.adaptive_density_control.append(
                    VanillaAdaptiveDensityController(
                        gaussians,
                        self._update_gauss_fn,
                        self.config.warmup_length,
                        self.config.refine_every,
                        self.config.reset_alpha_every,
                        self.config.stop_split_at,
                        self.config.continue_cull_post_densification,
                        self.num_train_data,
                        self.config.cull_alpha_thresh,
                        self.config.cull_scale_thresh,
                        self.config.densify_grad_thresh,
                        self.config.densify_size_thresh,
                        self.config.n_split_samples,
                        self.config.cull_screen_size,
                        self.config.split_screen_size,
                        self.config.stop_screen_size_at,
                        scene_box=scene_box,
                        max_gaussians=self.config.max_num_gaussians,
                        min_num_visible=self.config.adc_min_num_visible,
                        reset_tv_every=self.config.reset_tv_every,
                        reset_tv=True,
                    )
                )
            else:
                self.adaptive_density_control.append(
                    VanillaAdaptiveDensityController(
                        gaussians,
                        self._update_gauss_fn,
                        self.config.warmup_length,
                        self.config.refine_every,
                        self.config.reset_alpha_every,
                        self.config.stop_split_at,
                        self.config.continue_cull_post_densification,
                        self.num_train_data,
                        self.config.cull_alpha_thresh,
                        self.config.cull_scale_thresh,
                        self.config.densify_grad_thresh,
                        self.config.densify_size_thresh,
                        self.config.n_split_samples,
                        self.config.cull_screen_size,
                        self.config.split_screen_size,
                        self.config.stop_screen_size_at,
                        scene_box=scene_box,
                        max_gaussians=self.config.max_num_gaussians,
                        min_num_visible=self.config.adc_min_num_visible,
                    )
                )

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = lpips.LPIPS(net="alex")
        self.step = 0

        # define crop as scene_bounds
        self.crop_box: Optional[OrientedBox] = None

        # background
        if self.config.background_color == "random":
            self.background_color = torch.rand(3)
        else:
            self.background_color = get_color(self.config.background_color)

        camopt_kwargs = dict(device="cpu")
        if isinstance(self.config.camera_optimizer, VehiclePoseOptimizerConfig):
            camopt_kwargs["num_physical_cameras"] = self.num_cameras

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data + self.num_eval_data, **camopt_kwargs
        )

    def precompute_infer(self):
        pass

    def _update_gauss_fn(self, gaussians, idx):
        if idx == 0:
            self.scene_gaussians = gaussians
            return self.scene_gaussians
        elif idx == 1:
            self.object_gaussians = gaussians
            return self.object_gaussians
        else:
            raise ValueError("Unknown Gaussian index.")

    def _apply(self, fn):
        super()._apply(fn)
        self.scene_box.aabb = fn(self.scene_box.aabb)
        return self

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        self.scene_field.set_step(self.step)
        for head in self.dynamic_fields:
            head.set_step(self.step)

        # map config names to attribute names in Gaussian class
        gaussian_attr_names = ["means", "opacities", "scales", "quats", "object_ids", "temporal_vis", "point_life"]
        for name, gaussians in {"scene": self.scene_gaussians, "object": self.object_gaussians}.items():
            for p in gaussian_attr_names:
                if p not in ["means", "features_dc", "features_rest", "object_ids", "temporal_vis", "point_life"]:
                    p = f"_{p}"
                if not hasattr(gaussians, p):
                    continue
                newp = dict[f"{name}_gaussians.{p}"].shape[0]
                old_shape = getattr(gaussians, p).shape
                new_shape = (newp,) + old_shape[1:]
                setattr(gaussians, p, torch.nn.Parameter(torch.zeros(new_shape, device=self.device)))

        super().load_state_dict(dict, **kwargs)
        
    def after_train(self, step: int):
        assert step == self.step
        self.scene_field.set_step(step)
        for head in self.dynamic_fields:
            head.set_step(step)
        start_idx = 0
        visible_mask = (self.radii > 0).flatten()
        gauss_visible_mask = self.gaussian_mask.clone()
        infursm = self.gaussian_mask.clone()
        gauss_visible_mask[self.gaussian_mask] = visible_mask
        assert self.xys.absgrad is not None
        if self.config.temporal_vis:
            xy_grads_norm = self.xys.absgrad.detach().norm(dim=-1)[(self.temp_radii > 0).flatten()]
        else:
            xy_grads_norm = self.xys.absgrad.detach().norm(dim=-1)[visible_mask]
        all_radii = self.radii.detach()[visible_mask]
        if self.config.temporal_vis:
            self.scene_gaussians.update_vis(self.current_camera_time,self.current_camera_view, self.current_camera_seq,gauss_visible_mask[:self.scene_gaussians.num_points])
            # self.scene_gaussians.update_vis(self.current_camera_time,self.current_camera_view, self.current_camera_seq,infursm[:self.scene_gaussians.num_points])
        for adc in self.adaptive_density_control:
            adc.last_size = self.last_size
            end_idx = start_idx + adc.num_gaussians
            vis_mask = gauss_visible_mask[start_idx:end_idx]
            num_vis_sofar = gauss_visible_mask[:start_idx].sum()
            grads = xy_grads_norm[num_vis_sofar : num_vis_sofar + vis_mask.sum()]
            radii = all_radii[num_vis_sofar : num_vis_sofar + vis_mask.sum()]
            adc.update_statistics(step, vis_mask, grads, radii)
            start_idx = end_idx

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def get_background(self):
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        return background

    @torch.no_grad()
    def refinement_after(self, optimizers: Optimizers, pipeline, step: int):
        assert step == self.step
        stats = []
        n_bef = self.num_points
        for i, adc in enumerate(self.adaptive_density_control):
            adc.set_num_total_gaussians(n_bef)
            result = adc(self.step, i, optimizers)
            stats.append(result)

        refined = any(s is not None for s in stats)
        # if parameters have changed and we run in DDP, trigger reducer update
        if refined and isinstance(pipeline._model, torch.nn.parallel.DistributedDataParallel):
            trigger_reducer_update(pipeline._model)

        # print statistics
        if refined and get_rank() == 0:
            stats = list(filter(lambda x: x is not None, stats))
            num_splits, num_dups, num_culls = (sum(s[i] for s in stats) for i in range(3))
            CONSOLE.log(f"Splitting {num_splits/n_bef} gaussians: {num_splits}/{n_bef}")
            CONSOLE.log(f"Duplicating {num_dups/n_bef} gaussians: {num_dups}/{n_bef}")
            CONSOLE.log(f"Culled {num_culls} gaussians")
            CONSOLE.log(f"New number of gaussians: {self.num_points}")

    def _get_cls_per_obj(self):
        return [self.object_class_ids[self.object_ids == i][0].long() for i in range(int(self.object_ids.max()) + 1)]

    def _get_dim_per_obj(self):
        return [self.object_dims[self.object_ids == i][0] for i in range(int(self.object_ids.max()) + 1)]

    def _get_gaussians(self):
        if hasattr(self, "object_gaussians"):
            return [self.scene_gaussians, self.object_gaussians]
        else:
            return [self.scene_gaussians]

    @property
    def num_points(self):
        return sum(g.num_points for g in self._get_gaussians())
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers, training_callback_attributes.pipeline],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the gaussians."""
        params = {}
        for gaussians in self._get_gaussians():
            for k, v in gaussians.get_param_groups().items():
                if k in params:
                    params[k] += [v]
                else:
                    params[k] = [v]
        return params

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        # gaussians
        params = self.get_gaussian_param_groups()

        # field heads
        scene_head_params = list(self.scene_field.parameters())
        object_head_params = []
        for head in self.dynamic_fields:
            object_head_params.extend(list(head.parameters()))
        if len(scene_head_params) > 0:
            params["scene_head"] = scene_head_params
        if len(object_head_params) > 0:
            params["object_heads"] = object_head_params

        # scene level codes
        params["scene_embeddings"] = []
        if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
            params["scene_embeddings"] += list(self.scene_embedding_appearance.parameters())
        if self.sequence_ids.numel() > 1 and self.config.transient_embedding_dim > 0:
            params["scene_embeddings"] += list(self.scene_embedding_transient.parameters())
        if not len(params["scene_embeddings"]):
            del params["scene_embeddings"]

        # sensor level codes
        if self.config.sensor_appearance_embedding_dim > 0:
            params["sensor_embeddings"] = list(self.sensor_embedding_appearance.parameters())

        # camera optimizer param groups
        self.camera_optimizer.get_param_groups(param_groups=params)

        if self.config.optimize_box_poses:
            params["object_poses"] = [self.object_poses_delta]
        return params

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def _get_object_poses(self) -> Tensor:
        # apply delta for box param optim
        if not self.config.optimize_object_height:
            self.object_poses_delta[..., 2] *= 0.0
        object_poses = self.object_poses + self.object_poses_delta

        # add camera opt params with no grad. Can only be done if camera optimizer is VehiclePoseOptimizer
        if self.config.camera_optimizer.mode != "off" and isinstance(
            self.config.camera_optimizer, VehiclePoseOptimizerConfig
        ):
            all_pose_adjustment = self.camera_optimizer.pose_adjustment.detach()

            vehicle_poses = exp_map_SE3(all_pose_adjustment)
            object_poses[..., :3] = (vehicle_poses[:, None, :3, :3] @ object_poses[..., :3, None]).squeeze(
                -1
            ) + vehicle_poses[:, None, :3, 3]
            object_poses[..., 3] += torch.arctan2(vehicle_poses[..., 1, 0], vehicle_poses[..., 0, 0]).unsqueeze(-1)
        return object_poses
    
    def get_object_poses_time(self,time) -> Tensor:
        # apply delta for box param optim
        if not self.config.optimize_object_height:
            self.object_poses_delta[..., 2] *= 0.0
        object_poses = self.object_poses + self.object_poses_delta

        # add camera opt params with no grad. Can only be done if camera optimizer is VehiclePoseOptimizer
        if self.config.camera_optimizer.mode != "off" and isinstance(
            self.config.camera_optimizer, VehiclePoseOptimizerConfig
        ):
            print(1)
            all_pose_adjustment = self.camera_optimizer.pose_adjustment.detach()

            vehicle_poses = exp_map_SE3(all_pose_adjustment)
            object_poses[..., :3] = (vehicle_poses[:, None, :3, :3] @ object_poses[..., :3, None]).squeeze(
                -1
            ) + vehicle_poses[:, None, :3, 3]
            object_poses[..., 3] += torch.arctan2(vehicle_poses[..., 1, 0], vehicle_poses[..., 0, 0]).unsqueeze(-1)
        return object_poses

    def prepare_camera(self, camera: Cameras):
        """Prepare the camera for rendering (in-place). Applies camera optimizer if enabled."""
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            self.camera_optimizer.apply_to_camera(camera)

        # add viewer info
        if camera.metadata is None:
            camera.metadata = {}
            camera.times = torch.ones_like(camera.times, dtype=torch.float) * self.viewer_time.value
            if self.viewer_sequence is not None:
                camera.metadata["sequence_ids"] = (
                    torch.ones_like(camera.times, dtype=torch.long) * self.viewer_sequence.value
                )

        if "sequence_ids" not in camera.metadata:
            camera.metadata["sequence_ids"] = torch.zeros_like(camera.times, dtype=torch.long)

        if "camera_ids" not in camera.metadata:
            camera.metadata["camera_ids"] = torch.zeros_like(camera.times, dtype=torch.long)


    def get_camera_matrix(self, camera: Cameras,opengl=True):
        """Get the w2c matrix for rendering."""
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        if opengl:
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        else:
            R_edit = torch.diag(torch.tensor([1, 1, 1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        return viewmat
    
    def get_camera_matrix1(self, camera,opengl=True):
        """Get the w2c matrix for rendering."""
        # shift the camera to center of scene looking at center
        R = camera[:, :3, :3]  # 3 x 3
        T = camera[:, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        if opengl:
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))[None,...].repeat(R.shape[0],1,1)
        else:
            R_edit = torch.diag(torch.tensor([1, 1, 1], device=self.device, dtype=R.dtype))[None,...].repeat(R.shape[0],1,1)
        R1 = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R1.transpose(1, 2)
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)[None,...].repeat(R.shape[0],1,1)
        viewmat[:, :3, :3] = R_inv
        viewmat[:, :3, 3:4] = T_inv
        return viewmat

    def get_object_geometry(
        self, camera: Cameras, obj_ids: Tensor, obj_poses: Tensor, obj_dims_max: Tensor, obj_cls: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        time = camera.times[0]
        obj_id_matches = self.object_gaussians.object_ids == obj_ids
        mask = obj_id_matches.any(dim=1)
        obj_indices = obj_id_matches.max(dim=-1)[1]

        self.gaussian_mask[self.scene_gaussians.num_points :] = mask
        obj_poses = obj_poses[obj_indices]
        obj_poses = obj_poses[mask]
        obj_dims = obj_dims_max.unsqueeze(-1)[obj_indices]
        obj_dims = obj_dims[mask]
        obj_cls = obj_cls[obj_indices]
        obj_cls = obj_cls[mask]

        if self.num_classes > 1:
            obj_means = torch.zeros_like(obj_poses[:, :3])
            obj_scales = torch.zeros_like(obj_poses[:, :3])
            obj_quats = torch.zeros_like(obj_poses)
            obj_feats = {}
            for i in range(self.num_classes):
                mask_i = obj_cls == i
                if mask_i.any():  # if there are any Gaussians of class i in view frustum
                    mask_full = mask.clone()
                    mask_full[mask == True] = mask_i  # noqa: E712
                    means, scales, quats, feats = self.dynamic_fields[i].get_geometry(
                        self.object_gaussians, mask_full, time
                    )
                    obj_means[mask_i] = means
                    obj_scales[mask_i] = scales
                    obj_quats[mask_i] = quats
                    if feats is not None:
                        for k, v in feats.items():
                            if v is not None and k not in obj_feats:
                                obj_feats[k] = torch.zeros(
                                    (obj_poses.shape[0], *v.shape[1:]), dtype=v.dtype, device=v.device
                                )
                            if v is not None:
                                obj_feats[k][mask_i] = v
                obj_feats[f"mask_{i}"] = mask_i
        else:
            obj_means, obj_scales, obj_quats, obj_feats = self.dynamic_fields[0].get_geometry(
                self.object_gaussians, mask, time
            )

        # save a few precomputed results: canoncal coordinates, transformed cameras to local space
        if obj_feats is not None:
            local_c2w = c2w_to_local(
                camera.camera_to_worlds.repeat(obj_poses.shape[0], 1, 1), obj_poses.detach(), obj_dims
            )
            viewdirs = obj_means.detach() - local_c2w.detach()[..., :3, 3]  # (N, 3)
            obj_feats["viewdirs"] = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

        # transform gaussians (means, quats) to world space
        obj_means = obj_means * obj_dims
        # calculate world space scales given object dims in log space, i.e. ln(xy) = ln(x) + ln(y)
        obj_scales = obj_scales + torch.log(obj_dims)

        zeros = torch.zeros_like(obj_poses[:, 3])
        ones = torch.ones_like(obj_poses[:, 3])
        c, s = torch.cos(obj_poses[:, 3]), torch.sin(obj_poses[:, 3])
        R = torch.stack([c, -s, zeros, s, c, zeros, zeros, zeros, ones], -1).view(-1, 3, 3)
        obj_means = torch.einsum("...ij,...j->...i", R, obj_means) + obj_poses[:, :3]
        obj2world_quats = torch.stack(
            [torch.cos(obj_poses[:, 3] / 2), zeros, zeros, torch.sin(obj_poses[:, 3] / 2)], -1
        )
        obj_quats = quaternion_multiply(obj2world_quats, obj_quats)

        return obj_means, obj_scales, obj_quats, obj_feats

    def get_object_geometry_instance_proj(
        self, camera: Cameras, obj_ids: Tensor, obj_poses: Tensor, obj_dims_max: Tensor, obj_cls: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        time = camera.times[0]
        obj_id_matches = self.object_gaussians.object_ids == obj_ids
        mask = obj_id_matches.any(dim=1)
        obj_indices = obj_id_matches.max(dim=-1)[1]

        self.gaussian_mask[self.scene_gaussians.num_points :] = mask
        obj_poses1 = obj_poses[:]
        obj_dims1 = obj_dims_max.unsqueeze(-1)[:]
        obj_cls1 = obj_cls[:]
        obj_instance_ids = obj_indices[mask] + 1
        
        obj_poses = obj_poses[obj_indices]
        obj_poses = obj_poses[mask]
        obj_dims = obj_dims_max.unsqueeze(-1)[obj_indices]
        obj_dims = obj_dims[mask]
        obj_cls = obj_cls[obj_indices]
        obj_cls = obj_cls[mask]

        obj_viewmats = c2w_to_local_grad(
            camera.camera_to_worlds.repeat(obj_poses1.shape[0], 1, 1), obj_poses1, obj_dims1
        )
        
        if self.num_classes > 1:
            obj_means = torch.zeros_like(obj_poses[:, :3])
            obj_scales = torch.zeros_like(obj_poses[:, :3])
            obj_quats = torch.zeros_like(obj_poses)
            obj_feats = {}
            for i in range(self.num_classes):
                mask_i = obj_cls == i
                if mask_i.any():  # if there are any Gaussians of class i in view frustum
                    mask_full = mask.clone()
                    mask_full[mask == True] = mask_i  # noqa: E712
                    means, scales, quats, feats = self.dynamic_fields[i].get_geometry(
                        self.object_gaussians, mask_full, time
                    )
                    obj_means[mask_i] = means
                    obj_scales[mask_i] = scales
                    obj_quats[mask_i] = quats
                    if feats is not None:
                        for k, v in feats.items():
                            if v is not None and k not in obj_feats:
                                obj_feats[k] = torch.zeros(
                                    (obj_poses.shape[0], *v.shape[1:]), dtype=v.dtype, device=v.device
                                )
                            if v is not None:
                                obj_feats[k][mask_i] = v
                obj_feats[f"mask_{i}"] = mask_i
        else:
            obj_means, obj_scales, obj_quats, obj_feats = self.dynamic_fields[0].get_geometry(
                self.object_gaussians, mask, time
            )

        # save a few precomputed results: canoncal coordinates, transformed cameras to local space
        if obj_feats is not None:
            # local_c2w = c2w_to_local(
            #     camera.camera_to_worlds.repeat(obj_poses.shape[0], 1, 1), obj_poses.detach(), obj_dims
            # )
            local_c2w = obj_viewmats.clone().detach()
            local_c2w[:,:,3] = local_c2w[:,:,3] / obj_dims1
            local_c2w = local_c2w[obj_indices][mask]
            viewdirs = obj_means.detach() - local_c2w.detach()[..., :3, 3]  # (N, 3)
            obj_feats["viewdirs"] = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

        # transform gaussians (means, quats) to world space
        obj_means = obj_means * obj_dims
        # calculate world space scales given object dims in log space, i.e. ln(xy) = ln(x) + ln(y)
        obj_scales = obj_scales + torch.log(obj_dims)


        return obj_means, obj_scales, obj_quats, obj_feats, obj_viewmats, obj_instance_ids

    def get_colors_opacities(
        self, visible_mask: Tensor, camera: Cameras, obj_feats: Optional[Dict[str, Tensor]], scene_num: int = -1
    ) -> tuple[Tensor, Tensor]:
        time = camera.times[0]
        sequence = camera.metadata["sequence_ids"][0]
        camera_id = camera.metadata["camera_ids"][0]
        num_scene_points = self.scene_gaussians.num_points if scene_num == -1 else scene_num

        if self.sequence_ids.numel() > 1 and self.config.transient_embedding_dim > 0:
            geometry_embed = self.scene_embedding_transient(time, sequence)
        else:
            geometry_embed = None

        if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
            assert self.config.sensor_appearance_embedding_dim == 0
            appearance_embed = self.scene_embedding_appearance(time, sequence)
        elif self.config.sensor_appearance_embedding_dim > 0:
            appearance_embed = self.sensor_embedding_appearance(time, camera_id)
        else:
            appearance_embed = None

        colors_crop, opacities_crop = self.scene_field.get_colors_opacities(
            self.scene_gaussians, camera, visible_mask[:num_scene_points], None, geometry_embed, appearance_embed
        )

        num_visible = visible_mask[num_scene_points:].sum()
        if num_visible > 0:
            visible_obj_mask = self.gaussian_mask[num_scene_points:].clone()
            visible_obj_mask[visible_obj_mask == True] = visible_mask[num_scene_points:]  # noqa: E712

            if obj_feats is not None:
                obj_feats = {
                    k: v[visible_mask[num_scene_points:]] if isinstance(v, Tensor) else v for k, v in obj_feats.items()
                }

            if self.num_classes > 1:
                obj_colors = torch.zeros((num_visible, 3), device=self.device)
                obj_opacities = torch.zeros((num_visible, 1), device=self.device)
                for i in range(self.num_classes):
                    full_mask = visible_obj_mask.clone()
                    mask_i = obj_feats[f"mask_{i}"]
                    full_mask[full_mask == True] = mask_i  # noqa: E712
                    if full_mask.any():  # if there are any visible Gaussians for class i
                        if obj_feats is not None:
                            feats = {k: v[mask_i] for k, v in obj_feats.items()}
                        else:
                            feats = None
                        colors, opacities = self.dynamic_fields[i].get_colors_opacities(
                            self.object_gaussians, camera, full_mask, feats, None, appearance_embed
                        )
                        obj_colors[mask_i] = colors
                        obj_opacities[mask_i] = opacities
            else:
                obj_colors, obj_opacities = self.dynamic_fields[0].get_colors_opacities(
                    self.object_gaussians, camera, visible_obj_mask, obj_feats, None, appearance_embed
                )
            opacities_crop = torch.cat([opacities_crop, obj_opacities])
            colors_crop = torch.cat([colors_crop, obj_colors])

        return colors_crop, opacities_crop
    
    def get_colors_opacities_efficient(
        self, visible_mask: Tensor, camera: Cameras, obj_feats: Optional[Dict[str, Tensor]], add_noise_mask: Tensor, add_noise: Tensor, scene_num: int = -1
    ) -> tuple[Tensor, Tensor]:
        time1 = camera.times[0]
        
        sequence = camera.metadata["sequence_ids"][0]
        camera_id = camera.metadata["camera_ids"][0]
        num_scene_points = self.scene_gaussians.num_points if scene_num == -1 else scene_num

        if self.sequence_ids.numel() > 1 and self.config.transient_embedding_dim > 0:
            geometry_embed = self.scene_embedding_transient(time1, sequence)
        else:
            geometry_embed = None
        
        if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
            assert self.config.sensor_appearance_embedding_dim == 0
            appearance_embed = self.scene_embedding_appearance(time1, sequence)
        elif self.config.sensor_appearance_embedding_dim > 0:
            appearance_embed = self.sensor_embedding_appearance(time1, camera_id)
        else:
            appearance_embed = None
        
        colors_crop, opacities_crop = self.scene_field.get_colors_opacities_efficient(
            self.scene_gaussians, camera, visible_mask[:num_scene_points], None, geometry_embed, appearance_embed, add_noise_mask[:num_scene_points], add_noise[:num_scene_points]
        )
        
        num_visible = visible_mask[num_scene_points:].sum()
        if num_visible > 0:
            visible_obj_mask = self.gaussian_mask[num_scene_points:].clone()
            obj_add_noise = torch.zeros_like(self.object_gaussians.means)
            obj_add_noise[visible_obj_mask == True] = add_noise[num_scene_points:]
            visible_obj_mask[visible_obj_mask == True] = visible_mask[num_scene_points:]  # noqa: E712
            
            if obj_feats is not None:
                obj_feats = {
                    k: v[visible_mask[num_scene_points:]] if isinstance(v, Tensor) else v for k, v in obj_feats.items()
                }

            if self.num_classes > 1:
                obj_colors = torch.zeros((num_visible, 3), device=self.device)
                obj_opacities = torch.zeros((num_visible, 1), device=self.device)
                for i in range(self.num_classes):
                    full_mask = visible_obj_mask.clone()
                    mask_i = obj_feats[f"mask_{i}"]
                    full_mask[full_mask == True] = mask_i  # noqa: E712
                    if full_mask.any():  # if there are any visible Gaussians for class i
                        if obj_feats is not None:
                            feats = {k: v[mask_i] for k, v in obj_feats.items()}
                        else:
                            feats = None
                        colors, opacities = self.dynamic_fields[i].get_colors_opacities(
                            self.object_gaussians, camera, full_mask, feats, None, appearance_embed, add_noise_mask[num_scene_points:], obj_add_noise
                        )
                        obj_colors[mask_i] = colors
                        obj_opacities[mask_i] = opacities
            else:
                obj_colors, obj_opacities = self.dynamic_fields[0].get_colors_opacities(
                    self.object_gaussians, camera, visible_obj_mask, obj_feats, None, appearance_embed, add_noise_mask[num_scene_points:], obj_add_noise
                )
            opacities_crop = torch.cat([opacities_crop, obj_opacities])
            colors_crop = torch.cat([colors_crop, obj_colors])
        
        return colors_crop, opacities_crop
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs."""
        if not isinstance(camera, Cameras):
            raise ValueError("camera must be of type Cameras, got: ", type(camera))
        assert camera.shape[0] == 1, "Only one camera at a time"

        start_time = time.time()
        self.prepare_camera(camera)
        background = self.get_background()
        
        self.gaussian_mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        self.gaussian_mask[: self.scene_gaussians.num_points] = True
        
        if self.config.temporal_vis:
            self.current_camera_time = camera.times[0]
            self.current_camera_view = camera.metadata['camera_ids'][0]
            self.current_camera_seq = camera.metadata['sequence_ids'][0] 
            current_vis_mask = self.scene_gaussians.get_mask(camera.times[0], camera.metadata['camera_ids'][0], camera.metadata['sequence_ids'][0], train=True)
            ratio = (current_vis_mask.sum().float()/current_vis_mask.shape[0]).item()
            
            if ratio < 0.15:
                current_vis_mask = torch.ones_like(current_vis_mask)
            
            start_time = time.time()
            means_crop, scales_crop, quats_crop, _ = self.scene_field.get_geometry(self.scene_gaussians,mask=current_vis_mask)
        else:
            means_crop, scales_crop, quats_crop, _ = self.scene_field.get_geometry(self.scene_gaussians)
        instance_ids_crop = torch.zeros((means_crop.shape[0]),device=means_crop.device,dtype=torch.int)
        obj_feats = None
        start_time = time.time()
        if not self.config.is_static:
            # apply delta for box param optim
            
            time1 = camera.times[0]
            object_poses = self._get_object_poses()
            sequence = camera.metadata["sequence_ids"][0]
            # get objects at (s, t), interpolate poses to t
            
            obj_ids, obj_poses, obj_dims, obj_cls = get_objects_at_time(
                object_poses,
                self.object_ids,
                self.object_dims,
                self.object_times,
                self.object_seq_ids,
                self.object_class_ids,
                time1,
                sequence,
            )
            obj_dims_max = obj_dims.max(dim=-1)[0]
            # filter out objects that are out of view
            if self.config.nus:
                view_mask = frustum_check(obj_poses, obj_dims, camera)
            else:
                view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
            obj_ids = obj_ids[view_mask]
            obj_poses = obj_poses[view_mask]
            obj_dims = obj_dims[view_mask]
            obj_dims_max = obj_dims_max[view_mask]
            obj_cls = obj_cls[view_mask]
            
            if len(obj_ids) > 0:
                if self.config.instance_proj:
                    object_means, object_scales, object_quats, obj_feats, obj_viewmats, obj_instance_ids = self.get_object_geometry_instance_proj(
                        camera, obj_ids, obj_poses, obj_dims_max, obj_cls
                    )
                    instance_ids_crop = torch.cat([instance_ids_crop, obj_instance_ids])
                else:
                    object_means, object_scales, object_quats, obj_feats = self.get_object_geometry(
                        camera, obj_ids, obj_poses, obj_dims_max, obj_cls
                    )
                
                means_crop = torch.cat([means_crop, object_means])
                scales_crop = torch.cat([scales_crop, object_scales])
                quats_crop = torch.cat([quats_crop, object_quats])
        
        assert self.crop_box is None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        start_time = time.time()
        if self.config.instance_proj:
            from gsplat_efficient.project_gaussians import project_gaussians
            if (not self.config.is_static) and len(obj_ids) > 0:
                viewmat = self.get_camera_matrix1(torch.cat([camera.camera_to_worlds,obj_viewmats],dim=0),opengl=not self.config.nus)
                
            else:
                viewmat = self.get_camera_matrix1(camera.camera_to_worlds,opengl=not self.config.nus)
                
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                means_crop,
                instance_ids_crop.to(torch.int32),
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat[:, :3, :],
                camera.fx.item(),
                camera.fy.item(),
                camera.cx.item(),
                camera.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
                clip_thresh=self.config.near_clip_thresh * self.dataparser_scale,
            )  # type: ignore
        else:
            from gsplat.project_gaussians import project_gaussians
            viewmat = self.get_camera_matrix(camera,opengl=not self.config.nus)
            # viewmat = torch.linalg.inv(camera.camera_to_worlds)
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                camera.fx.item(),
                camera.fy.item(),
                camera.cx.item(),
                camera.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
                clip_thresh=self.config.near_clip_thresh * self.dataparser_scale,
            )  # type: ignore

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation}
        
        if self.training and self.xys.requires_grad:
            self.xys.retain_grad()
        start_time = time.time()
        # get opacities, colors for the Gaussians that hit a tile
        if self.config.temporal_vis:
            vis_scene_num = current_vis_mask.sum()
            visible_mask = torch.zeros((current_vis_mask.shape[0]+self.radii[vis_scene_num:].shape[0]),device=self.radii.device,dtype=torch.bool)
            temp_radii = torch.zeros((current_vis_mask.shape[0]+self.radii[vis_scene_num:].shape[0]),device=self.radii.device,dtype=self.radii.dtype)
            temp_visible_mask = self.radii > 0
            visible_mask[:current_vis_mask.shape[0]][current_vis_mask] = temp_visible_mask[:vis_scene_num]
            visible_mask[current_vis_mask.shape[0]:] = temp_visible_mask[vis_scene_num:]
            temp_radii[:current_vis_mask.shape[0]][current_vis_mask] = self.radii[:vis_scene_num]
            temp_radii[current_vis_mask.shape[0]:] = self.radii[vis_scene_num:]
        else:
            visible_mask = self.radii > 0
        colors_crop = torch.zeros_like(conics)
        opacities_crop = torch.zeros_like(conics[:, :1])
        if self.config.temporal_vis:
            if self.config.autolod:
                
                radii_mask = (0 < temp_radii) & (temp_radii <= self.config.lod)
                radii_mask_w_t = (0 < self.radii) & (self.radii <= self.config.lod)
                drop_prob =  torch.zeros_like(radii_mask).float()
                drop_prob[radii_mask] = P_real(depths[radii_mask_w_t]/self.dataparser_scale, self.config.max_dis)
                add_noise_mask = torch.zeros_like(radii_mask)
                add_noise_mask[radii_mask] = torch.bernoulli(1 - drop_prob[radii_mask]) > 0.5
                drop_mask = radii_mask.clone()
                drop_mask[add_noise_mask] = False
                temp_visible_mask[:vis_scene_num][drop_mask[:current_vis_mask.shape[0]][current_vis_mask]] = False
                temp_visible_mask[vis_scene_num:][drop_mask[current_vis_mask.shape[0]:]] = False
                visible_mask[drop_mask] = False
                add_noise = drop_prob[...,None].repeat(1,3) * 0.05 * (torch.rand((add_noise_mask.shape[0],3),device=visible_mask.device) * 2 - 1)
                
                start_time = time.time()
                colors_crop[temp_visible_mask], opacities_crop[temp_visible_mask] = self.get_colors_opacities_efficient(
                    visible_mask, camera, obj_feats, add_noise_mask, add_noise
                )
            else:
                colors_crop[temp_visible_mask], opacities_crop[temp_visible_mask] = self.get_colors_opacities(
                    visible_mask, camera, obj_feats
                )
        elif self.training and self.config.autolod:
            radii_mask = (0 < self.radii) & (self.radii <= self.config.lod)
            
            drop_prob =  torch.zeros_like(radii_mask).float()
            drop_prob[radii_mask] = P_real(depths[radii_mask]/self.dataparser_scale, self.config.max_dis)
            add_noise_mask = torch.zeros_like(radii_mask)
            add_noise_mask[radii_mask] = torch.bernoulli(1 - drop_prob[radii_mask]) > 0.5
            drop_mask = radii_mask.clone()
            drop_mask[add_noise_mask] = False
            visible_mask[drop_mask] = False
            add_noise = drop_prob[...,None].repeat(1,3) * 0.05 * (torch.rand((add_noise_mask.shape[0],3),device=visible_mask.device) * 2 - 1)
            colors_crop[visible_mask], opacities_crop[visible_mask] = self.get_colors_opacities_efficient(
                visible_mask, camera, obj_feats, add_noise_mask, add_noise
            )
        else:
            colors_crop[visible_mask], opacities_crop[visible_mask] = self.get_colors_opacities(
                visible_mask, camera, obj_feats
            )

        
        # anti aliasing
        opacities_crop = torch.sigmoid(opacities_crop) * comp[:, None]

        # Clone the camera
        original_camera = camera._apply_fn_to_fields(lambda x: x.clone())

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        assert visible_mask.any()  # type: ignore
        start_time = time.time()
        rgb, alpha = rasterize_gaussians(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            colors_crop,
            opacities_crop,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore

        alpha = alpha.unsqueeze(-1)
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        out = {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "camera": camera,}  # type: ignore

        if self.config.depth_loss_mult > 0 or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                opacities_crop,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]
            out["depth"] = torch.where(alpha > 1e-6, depth_im / alpha, depth_im.detach().max())

        if not self.training and self.render_images:
            with torch.no_grad():
                # Visualize gaussians (downscaled)
                rgb_ctrl, _ = rasterize_gaussians(
                    self.xys,
                    depths,
                    self.radii,
                    conics * 10,
                    num_tiles_hit,
                    colors_crop.float(),
                    opacities_crop,
                    H,
                    W,
                    BLOCK_WIDTH,
                    background=torch.zeros(3, device=self.device),
                    return_alpha=True,
                )  # type: ignore
                rgb_ctrl.clamp_max_(1.0)

                # Add control points
                if not self.config.is_static:
                    # Add 3D boxes
                    rgb_ctrl = draw_boxes_in_image(
                        rgb_ctrl,
                        camera.camera_to_worlds[0],
                        camera.get_intrinsics_matrices()[0],
                        obj_poses,
                        obj_dims,
                        obj_ids,
                    )
                    ctrl_xyz = means_crop[self.scene_gaussians.num_points :]
                    ctrl_color = torch.zeros_like(ctrl_xyz)
                    ctrl_color[:, 0] = 255
                    points_raster = rasterize_points(
                        original_camera, ctrl_xyz, ctrl_color, out=rgb_ctrl.cpu(), marker_size=2
                    )
                    out["control_points"] = points_raster.to(self.device)
                
        if self.config.temporal_vis:
            self.temp_radii = self.radii.clone()
            self.radii = temp_radii.clone()
            
        
        return out

    def get_outputs_infer(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs."""
        if not isinstance(camera, Cameras):
            raise ValueError("camera must be of type Cameras, got: ", type(camera))
        assert camera.shape[0] == 1, "Only one camera at a time"

        start_time = time.time()
        self.prepare_camera(camera)
        background = self.get_background()
        get_background_time = time.time() - start_time
        start_time = time.time()
        self.gaussian_mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        self.gaussian_mask[: self.scene_gaussians.num_points] = True
        
        if self.config.temporal_vis:
            self.current_camera_time = camera.times[0]
            self.current_camera_view = camera.metadata['camera_ids'][0]
            self.current_camera_seq = camera.metadata['sequence_ids'][0] 
            current_vis_mask = self.scene_gaussians.get_mask(camera.times[0], camera.metadata['camera_ids'][0], camera.metadata['sequence_ids'][0], train=True)
            # ratio = (current_vis_mask.sum().float()/current_vis_mask.shape[0]).item()
            # print(ratio)
            # if ratio < 0.15:
            #     current_vis_mask = torch.ones_like(current_vis_mask)
            # # wandb.log({"mask ratio": ratio})
            get_temporal_time = time.time() - start_time
            start_time = time.time()
            means_crop, scales_crop, quats_crop, _ = self.scene_field.get_geometry(self.scene_gaussians,mask=current_vis_mask)
        else:
            means_crop, scales_crop, quats_crop, _ = self.scene_field.get_geometry(self.scene_gaussians)
        instance_ids_crop = torch.zeros((means_crop.shape[0]),device=means_crop.device,dtype=torch.int)
        obj_feats = None
        
        
        # if self.config.temporal_vis:
        #     self.sequence_ids
        start_time = time.time()
        if not self.config.is_static:
            # apply delta for box param optim
            
            time1 = camera.times[0]
            object_poses = self._get_object_poses()
            sequence = camera.metadata["sequence_ids"][0]
            # get objects at (s, t), interpolate poses to t
            
            obj_ids, obj_poses, obj_dims, obj_cls = get_objects_at_time(
                object_poses,
                self.object_ids,
                self.object_dims,
                self.object_times,
                self.object_seq_ids,
                self.object_class_ids,
                time1,
                sequence,
            )
            obj_dims_max = obj_dims.max(dim=-1)[0]
            # filter out objects that are out of view
            if self.config.nus:
                view_mask = frustum_check(obj_poses, obj_dims, camera)
            else:
                view_mask = opengl_frustum_check(obj_poses, obj_dims, camera)
            obj_ids = obj_ids[view_mask]
            obj_poses = obj_poses[view_mask]
            obj_dims = obj_dims[view_mask]
            obj_dims_max = obj_dims_max[view_mask]
            obj_cls = obj_cls[view_mask]
            obj_get_time = time.time() - start_time
            # transform object points to world space at time t, add to scene points
            start_time = time.time()
            if len(obj_ids) > 0:
                if self.config.instance_proj:
                    object_means, object_scales, object_quats, obj_feats, obj_viewmats, obj_instance_ids = self.get_object_geometry_instance_proj(
                        camera, obj_ids, obj_poses, obj_dims_max, obj_cls
                    )
                    instance_ids_crop = torch.cat([instance_ids_crop, obj_instance_ids])
                else:
                    object_means, object_scales, object_quats, obj_feats = self.get_object_geometry(
                        camera, obj_ids, obj_poses, obj_dims_max, obj_cls
                    )
                # print(object_means.max())
                means_crop = torch.cat([means_crop, object_means])
                scales_crop = torch.cat([scales_crop, object_scales])
                quats_crop = torch.cat([quats_crop, object_quats])
            obj_pre_time = time.time() - start_time
        
        assert self.crop_box is None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        start_time = time.time()
        if self.config.instance_proj:
            from gsplat_efficient.project_gaussians import project_gaussians
            if (not self.config.is_static) and len(obj_ids) > 0:
                viewmat = self.get_camera_matrix1(torch.cat([camera.camera_to_worlds,obj_viewmats],dim=0),opengl=not self.config.nus)
                # viewmat = torch.linalg.inv(camera.camera_to_worlds)
            else:
                viewmat = self.get_camera_matrix1(camera.camera_to_worlds,opengl=not self.config.nus)
                # viewmat = torch.linalg.inv(camera.camera_to_worlds)
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                means_crop,
                instance_ids_crop.to(torch.int32),
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat[:, :3, :],
                camera.fx.item(),
                camera.fy.item(),
                camera.cx.item(),
                camera.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
                clip_thresh=self.config.near_clip_thresh * self.dataparser_scale,
            )  # type: ignore
        else:
            from gsplat.project_gaussians import project_gaussians
            viewmat = self.get_camera_matrix(camera,opengl=not self.config.nus)
            # viewmat = torch.linalg.inv(camera.camera_to_worlds)
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                camera.fx.item(),
                camera.fy.item(),
                camera.cx.item(),
                camera.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
                clip_thresh=self.config.near_clip_thresh * self.dataparser_scale,
            )  # type: ignore

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation}
        # wandb.log({"3D Gaussian projection": time.time() - start_time})
        proj_gaus_time = time.time() - start_time
        # Important to allow xys grads to populate properly
        if self.training and self.xys.requires_grad:
            self.xys.retain_grad()
        start_time = time.time()
        # get opacities, colors for the Gaussians that hit a tile
        if self.config.temporal_vis:
            vis_scene_num = current_vis_mask.sum()
            visible_mask = torch.zeros((current_vis_mask.shape[0]+self.radii[vis_scene_num:].shape[0]),device=self.radii.device,dtype=torch.bool)
            temp_radii = torch.zeros((current_vis_mask.shape[0]+self.radii[vis_scene_num:].shape[0]),device=self.radii.device,dtype=self.radii.dtype)
            temp_visible_mask = self.radii > 0
            visible_mask[:current_vis_mask.shape[0]][current_vis_mask] = temp_visible_mask[:vis_scene_num]
            visible_mask[current_vis_mask.shape[0]:] = temp_visible_mask[vis_scene_num:]
            temp_radii[:current_vis_mask.shape[0]][current_vis_mask] = self.radii[:vis_scene_num]
            temp_radii[current_vis_mask.shape[0]:] = self.radii[vis_scene_num:]
        else:
            visible_mask = self.radii > 0
        colors_crop = torch.zeros_like(conics)
        opacities_crop = torch.zeros_like(conics[:, :1])
        if self.config.temporal_vis:
            if self.config.autolod:
                
                radii_mask = (0 < temp_radii) & (temp_radii <= self.config.lod)
                radii_mask_w_t = (0 < self.radii) & (self.radii <= self.config.lod)
                drop_prob =  torch.zeros_like(radii_mask).float()
                drop_prob[radii_mask] = P_real(depths[radii_mask_w_t]/self.dataparser_scale, self.config.max_dis)
                add_noise_mask = torch.zeros_like(radii_mask)
                add_noise_mask[radii_mask] = torch.bernoulli(1 - drop_prob[radii_mask]) > 0.5
                drop_mask = radii_mask.clone()
                drop_mask[add_noise_mask] = False
                temp_visible_mask[:vis_scene_num][drop_mask[:current_vis_mask.shape[0]][current_vis_mask]] = False
                temp_visible_mask[vis_scene_num:][drop_mask[current_vis_mask.shape[0]:]] = False
                visible_mask[drop_mask] = False
                add_noise = drop_prob[...,None].repeat(1,3) * 0.05 * (torch.rand((add_noise_mask.shape[0],3),device=visible_mask.device) * 2 - 1)
                get_autoLOD = time.time() - start_time
                start_time = time.time()
                colors_crop[temp_visible_mask], opacities_crop[temp_visible_mask] = self.get_colors_opacities_efficient(
                    visible_mask, camera, obj_feats, add_noise_mask, add_noise
                )
            else:
                colors_crop[temp_visible_mask], opacities_crop[temp_visible_mask] = self.get_colors_opacities(
                    visible_mask, camera, obj_feats
                )
        elif self.training and self.config.autolod:
            # radii_mask = (0 < num_tiles_hit[:self.scene_gaussians.num_points]) & (num_tiles_hit[:self.scene_gaussians.num_points] <= self.config.lod)
            radii_mask = (0 < self.radii) & (self.radii <= self.config.lod)
            drop_prob =  torch.zeros_like(radii_mask).float()
            drop_prob[radii_mask] = P_real(depths[radii_mask]/self.dataparser_scale, self.config.max_dis)
            add_noise_mask = torch.zeros_like(radii_mask)
            add_noise_mask[radii_mask] = torch.bernoulli(1 - drop_prob[radii_mask]) > 0.5
            drop_mask = radii_mask.clone()
            drop_mask[add_noise_mask] = False
            visible_mask[drop_mask] = False
            add_noise = drop_prob[...,None].repeat(1,3) * 0.05 * (torch.rand((add_noise_mask.shape[0],3),device=visible_mask.device) * 2 - 1)
            colors_crop[visible_mask], opacities_crop[visible_mask] = self.get_colors_opacities_efficient(
                visible_mask, camera, obj_feats, add_noise_mask, add_noise
            )
        else:
            colors_crop[visible_mask], opacities_crop[visible_mask] = self.get_colors_opacities(
                visible_mask, camera, obj_feats
            )

        query_neural_fields = time.time() - start_time
        # anti aliasing
        opacities_crop = torch.sigmoid(opacities_crop) * comp[:, None]

        # Clone the camera
        original_camera = camera._apply_fn_to_fields(lambda x: x.clone())

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        assert visible_mask.any()  # type: ignore
        start_time = time.time()
        rgb, alpha = rasterize_gaussians(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            colors_crop,
            opacities_crop,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        # wandb.log({"Rasterization time": time.time() - start_time})
        render_gaus_time = time.time() - start_time
        alpha = alpha.unsqueeze(-1)
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        out = {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "camera": camera,}  # type: ignore

        if self.config.depth_loss_mult > 0 or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                opacities_crop,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]
            out["depth"] = torch.where(alpha > 1e-6, depth_im / alpha, depth_im.detach().max())

        if not self.training and self.render_images:
            with torch.no_grad():
                # Visualize gaussians (downscaled)
                rgb_ctrl, _ = rasterize_gaussians(
                    self.xys,
                    depths,
                    self.radii,
                    conics * 10,
                    num_tiles_hit,
                    colors_crop.float(),
                    opacities_crop,
                    H,
                    W,
                    BLOCK_WIDTH,
                    background=torch.zeros(3, device=self.device),
                    return_alpha=True,
                )  # type: ignore
                rgb_ctrl.clamp_max_(1.0)

                # Add control points
                if not self.config.is_static:
                    # Add 3D boxes
                    rgb_ctrl = draw_boxes_in_image(
                        rgb_ctrl,
                        camera.camera_to_worlds[0],
                        camera.get_intrinsics_matrices()[0],
                        obj_poses,
                        obj_dims,
                        obj_ids,
                    )
                    ctrl_xyz = means_crop[self.scene_gaussians.num_points :]
                    ctrl_color = torch.zeros_like(ctrl_xyz)
                    ctrl_color[:, 0] = 255
                    points_raster = rasterize_points(
                        original_camera, ctrl_xyz, ctrl_color, out=rgb_ctrl.cpu(), marker_size=2
                    )
                    out["control_points"] = points_raster.to(self.device)
        if self.config.temporal_vis:
            self.temp_radii = self.radii.clone()
            self.radii = temp_radii.clone()
            
        return out

    def _get_predicted_and_gt_rgb(self, outputs, batch, do_mask_images: bool = True):
        """Get the predicted and ground truth RGB images, potentially downscaled."""
        d = self._get_downscale_factor()
        gt_img = batch["image"]

        # If stored as uint8, convert to float, divide by 255
        if gt_img.dtype == torch.uint8:
            gt_img = gt_img.float() / 255

        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        if d > 1:
            gt_rgb = resize_image(gt_rgb, d)

        # mask images
        if do_mask_images and not self.config.is_dynamic_mask:
            gt_rgb, predicted_rgb = mask_images(batch, gt_rgb, predicted_rgb, size_divisor=d)

        assert gt_rgb.shape == predicted_rgb.shape, f"Shapes do not match: {gt_rgb.shape} vs {predicted_rgb.shape}"
        return predicted_rgb, gt_rgb

    def _get_predicted_and_gt_depth(self, outputs, batch, do_mask_images: bool = True):
        """Get the predicted and ground truth depth images, potentially downscaled."""
        d = self._get_downscale_factor()
        gt_depth = batch["depth_image"].to(self.device)
        predicted_depth = outputs["depth"]

        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_depth = F.interpolate(gt_depth.permute(2, 0, 1).unsqueeze(0), newsize, mode="nearest")
            gt_depth = gt_depth.squeeze(0).permute(1, 2, 0)

        # mask images
        if do_mask_images and not self.config.is_dynamic_mask:
            gt_depth, predicted_depth = mask_images(batch, gt_depth, predicted_depth, size_divisor=d)
        return predicted_depth, gt_depth

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        predicted_rgb, gt_rgb = self._get_predicted_and_gt_rgb(outputs, batch)

        metrics_dict["psnr"] = psnr(predicted_rgb, gt_rgb)

        # neural embedding statistics
        if self.training:
            # scene embeddings
            if self.sequence_ids.numel() > 1 and self.config.appearance_embedding_dim > 0:
                app_std, app_mean = torch.std_mean(self.scene_embedding_appearance.sequence_code_weights)
                metrics_dict["scene_embedding_appearance_mean"] = app_mean
                metrics_dict["scene_embedding_appearance_stddev"] = app_std

            if self.sequence_ids.numel() > 1 and self.config.transient_embedding_dim > 0:
                trans_std, trans_mean = torch.std_mean(self.scene_embedding_transient.sequence_code_weights)
                metrics_dict["scene_embedding_transient_mean"] = trans_mean
                metrics_dict["scene_embedding_transient_stddev"] = trans_std

            if self.config.sensor_appearance_embedding_dim > 0:
                sensor_std, sensor_mean = torch.std_mean(self.sensor_embedding_appearance.sequence_code_weights)
                metrics_dict["sensor_embedding_mean"] = sensor_mean
                metrics_dict["sensor_embedding_stddev"] = sensor_std

        # object refinement magnitude
        if self.config.optimize_box_poses:
            metrics_dict["object_pose_delta_t"] = (
                self.object_poses_delta[..., :3].norm(dim=-1).mean() / self.dataparser_scale
            )
            metrics_dict["object_pose_delta_r"] = torch.rad2deg(self.object_poses_delta[..., 3]).abs().mean()

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        metrics_dict["gaussian_count"] = self.scene_gaussians.num_points
        if hasattr(self, "object_gaussians"):
            metrics_dict["object_gaussian_count"] = self.object_gaussians.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        predicted_rgb, gt_rgb = self._get_predicted_and_gt_rgb(outputs, batch)

        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])

        depth_loss = 0.0
        if outputs["depth"] is not None and self.config.depth_loss_mult > 0:
            depth, gt_depth = self._get_predicted_and_gt_depth(outputs, batch)
            mask = gt_depth > 0
            if mask.any():
                depth_loss = (gt_depth[mask] - depth[mask]).abs().mean() * self.config.depth_loss_mult

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "depth_loss": depth_loss,
        }

        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        # print(self.config.instance_proj, self.config.temporal_vis, self.config.autolod)
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        if camera.metadata is not None:
            self.render_images = camera.metadata.get("render_images", True)
            if "render_images" in camera.metadata:
                camera.metadata.pop("render_images")
        outs = self.get_outputs_infer(camera.to(self.device))
        self.render_images = False
        return outs  # type: ignore

    def _compute_metrics(self, predicted_rgb, gt_rgb):
        psnr_val = psnr(gt_rgb, predicted_rgb)
        ssim = ssim_fn(gt_rgb[None], predicted_rgb[None])
        gt_rgb = gt_rgb.permute(2, 0, 1).unsqueeze(0)
        predicted_rgb = predicted_rgb.permute(2, 0, 1).unsqueeze(0)
        lpips_val = self.lpips(gt_rgb, predicted_rgb, normalize=True)
        return psnr_val, ssim, lpips_val

    def _compute_masked_metrics(self, predicted_rgb, gt_rgb, mask, size_divisor: int = 1):
        if size_divisor > 1:
            newsize = [mask.shape[0] // size_divisor, mask.shape[1] // size_divisor]
            mask = F.interpolate(mask.permute(2, 0, 1).unsqueeze(0), newsize, mode="nearest")
            mask = mask.squeeze(0).permute(1, 2, 0)
        # invert boolen mask to only compute metrics on the masked area
        mask = (~mask).squeeze(-1)
        if not mask.any():
            # return nan so that the values will be filtered when computing the average
            return torch.nan, torch.nan

        psnr_val = psnr(gt_rgb[mask], predicted_rgb[mask])
        ssim = ssim_fn(gt_rgb[None], predicted_rgb[None], reduction="none")
        ssim = ssim.squeeze(0).permute(1, 2, 0)[mask].mean()
        return psnr_val, ssim

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        d = self._get_downscale_factor()
        predicted_rgb, gt_rgb = self._get_predicted_and_gt_rgb(outputs, batch, do_mask_images=False)
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # mask images if not dynamic mask
        if not self.config.is_dynamic_mask:
            gt_rgb, predicted_rgb = mask_images(batch, gt_rgb, predicted_rgb, size_divisor=d)

        # full image metrics
        psnr_val, ssim, lpips_val = self._compute_metrics(predicted_rgb, gt_rgb)
        psnr_perpixel = psnr(gt_rgb, predicted_rgb, reduction="none").mean(dim=-1, keepdim=True).clamp(0.0, 100.0)
        psnr_perpixel = (psnr_perpixel - psnr_perpixel.min()) / (psnr_perpixel.max() - psnr_perpixel.min())
        metrics_dict = {"psnr": psnr_val, "ssim": ssim, "lpips": lpips_val}

        # dynamic area metrics (if applicable)
        if self.config.is_dynamic_mask:
            psnr_val, ssim = self._compute_masked_metrics(predicted_rgb, gt_rgb, batch["mask"], size_divisor=d)
            metrics_dict["psnr_dynamic"] = psnr_val
            metrics_dict["ssim_dynamic"] = ssim

        acc = colormaps.apply_colormap(outputs["accumulation"])
        psnr_perpixel = colormaps.apply_colormap(psnr_perpixel)
        images_dict = {"img": combined_rgb, "accumulation": acc, "psnr_heatmap": psnr_perpixel}
        if "depth" in outputs and outputs["depth"] is not None:
            depth = apply_depth_colormap(
                outputs["depth"],
                self.config.min_depth * self.dataparser_scale,
                self.config.max_depth * self.dataparser_scale,
            )
            images_dict["depth"] = depth

        if "control_points" in outputs:
            images_dict["control_points"] = outputs["control_points"]

        if "boxes_bev" in outputs:
            images_dict["boxes_bev"] = outputs["boxes_bev"]

        return metrics_dict, images_dict

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle):
        """Get outputs for a camera ray bundle."""
        camera = Cameras(
            camera_to_worlds=camera_ray_bundle.camera_to_worlds,
            fx=camera_ray_bundle.fx,
            fy=camera_ray_bundle.fy,
            cx=camera_ray_bundle.cx,
            cy=camera_ray_bundle.cy,
            width=camera_ray_bundle.width,
            height=camera_ray_bundle.height,
            times=camera_ray_bundle.times,
            metadata=camera_ray_bundle.metadata,
        )
        return self.get_outputs(camera)

    def _downsample_img(self, img: torch.Tensor, out_size: Tuple[int, int] = (299, 299)):
        """Converts tensor to PIL, downsamples with bicubic, and converts back to tensor."""
        img = to_pil_image(img)
        img = img.resize(out_size, Image.BICUBIC)
        img = to_tensor(img)
        return img

    def _update_lane_shift_fid(self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img, gen_img):
        """Updates the FID metrics (for shifted views) for the given camera and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {0, 2, 3}, "Shift amounts are hardcoded for now."
        imgs_generated = {0: gen_img}

        # Calculate shift direction based on driving direction or default to x-axis
        if "velocities" in camera.metadata:
            # print(camera.metadata["velocities"])
            driving_direction = camera.metadata["velocities"][0, :].cuda()
            driving_direction = driving_direction / driving_direction.norm()
        else:
            # Default to x-axis direction if no velocity information
            driving_direction = torch.tensor([1.0, 0.0, 0.0]).cuda()
        orth_right_direction = driving_direction #torch.cross(
        #     driving_direction, torch.tensor([0.0, 0.0, 1.0], device=driving_direction.device)
        # )

        # Create output directory if it doesn't exist
        output_dir = Path("fid_renders")
        output_dir.mkdir(exist_ok=True)
        cam_idx = camera.metadata["cam_idx"][0][0].item()
        # print(cam_idx)
        # assert 1 == 2
        # Save original image
        Image.fromarray((orig_img.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"original_{cam_idx}.jpg")
        Image.fromarray((gen_img.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"generated_0_{cam_idx}.jpg")

        # Apply shifts and save images
        # print(camera.camera_to_worlds[0, :3, 3])
        original_camera = copy.deepcopy(camera)
        original_camera.camera_to_worlds[..., :2, 3] += 2 * orth_right_direction.cpu()[:2] * self.dataparser_scale * 0.5
        img_2 = self.get_outputs_for_camera(original_camera)["rgb"]
        imgs_generated[2] = img_2
        Image.fromarray((img_2.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"generated_2_{cam_idx}.jpg")
        # print(camera.camera_to_worlds[0, :3, 3])
        original_camera = copy.deepcopy(camera)
        original_camera.camera_to_worlds[..., :2, 3] += 3 * orth_right_direction.cpu()[:2] * self.dataparser_scale * 0.5
        img_3 = self.get_outputs_for_camera(original_camera)["rgb"]
        imgs_generated[3] = img_3
        Image.fromarray((img_3.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"generated_3_{cam_idx}.jpg")

        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def _update_vertical_shift_fid(self, fids: Dict[int, FrechetInceptionDistance], camera, orig_img):
        """Updates the FID metrics (for shifted views) for the given camera and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # Compute FID for shifted views
        assert fids.keys() == {1}, "Shift amounts are hardcoded for now."
        imgs_generated = {}

        # Create output directory if it doesn't exist
        output_dir = Path("fid_renders")
        output_dir.mkdir(exist_ok=True)
        cam_idx = camera.metadata["cam_idx"][0][0].item()
        # Save original image
        Image.fromarray((orig_img.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"vertical_original_{cam_idx}.jpg")

        # Apply vertical shift and save image
        original_camera = copy.deepcopy(camera)
        original_camera.camera_to_worlds[..., 2, 3] += self.dataparser_scale * 0.5
        img_1 = self.get_outputs_for_camera(original_camera)["rgb"]
        imgs_generated[1] = img_1
        Image.fromarray((img_1.cpu().numpy() * 255).astype(np.uint8)).save(output_dir / f"vertical_generated_1_{cam_idx}.jpg")

        for shift, img in imgs_generated.items():
            img = (self._downsample_img((img).permute(2, 0, 1)) * 255).unsqueeze(0).to(torch.uint8).to(self.device)
            fids[shift].update(img, real=False)

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and compute the average metrics."""
        metrics_list = []
        assert isinstance(self.eval_dataset, InputDataset)
        for camera_idx in range(len(self.eval_dataset)):
            camera = self.eval_dataset.cameras[camera_idx]
            image_idx = self.eval_dataset.image_indices[camera_idx]
            image = self.eval_dataset.images[image_idx]
            outputs = self.get_outputs_for_camera(camera)
            metrics, _ = self.get_image_metrics_and_images(outputs, image)
            metrics_list.append(metrics)

            # Calculate FID metrics if needed
            if step in self.config.calc_fid_steps:
                # Initialize FID metrics if not already done
                if not hasattr(self, "fid_metrics"):
                    self.fid_metrics = {
                        "lane_shift": {
                            0: FrechetInceptionDistance(normalize=True).to(self.device),
                            2: FrechetInceptionDistance(normalize=True).to(self.device),
                            3: FrechetInceptionDistance(normalize=True).to(self.device),
                        },
                        "vertical_shift": {
                            0: FrechetInceptionDistance(normalize=True).to(self.device),
                            1: FrechetInceptionDistance(normalize=True).to(self.device),
                        },
                    }

                # Update FID metrics
                self._update_lane_shift_fid(
                    self.fid_metrics["lane_shift"],
                    camera,
                    image,
                    outputs["rgb"],
                )
                self._update_vertical_shift_fid(
                    self.fid_metrics["vertical_shift"],
                    camera,
                    image,
                )

        # Average metrics across all images
        metrics = {}
        for key in metrics_list[0].keys():
            metrics[key] = float(
                torch.mean(torch.stack([m[key] for m in metrics_list]))
            )

        # Add FID metrics if calculated
        if step in self.config.calc_fid_steps:
            metrics["fid_lane_shift_0"] = self.fid_metrics["lane_shift"][0].compute().item()
            metrics["fid_lane_shift_2"] = self.fid_metrics["lane_shift"][2].compute().item()
            metrics["fid_lane_shift_3"] = self.fid_metrics["lane_shift"][3].compute().item()
            metrics["fid_vertical_shift_0"] = self.fid_metrics["vertical_shift"][0].compute().item()
            metrics["fid_vertical_shift_1"] = self.fid_metrics["vertical_shift"][1].compute().item()

        return metrics

