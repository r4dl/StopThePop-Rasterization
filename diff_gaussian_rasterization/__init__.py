#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import dacite
from dataclasses import dataclass, asdict, field
from typing import NamedTuple
import torch.nn as nn
import torch
from enum import IntEnum
from . import _C

def enum_dict_factory(data):
    def convert_value(obj):
        if isinstance(obj, IntEnum):
            return obj.value
        return obj
    return dict((k, convert_value(v)) for k, v in data)
    
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.inv_viewprojmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.settings.to_dict(),
            raster_settings.render_depth,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, opacities, scales, rotations, cov3Ds_precomp, radii, sh, color, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, opacities, scales, rotations, cov3Ds_precomp, radii, sh, color, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                opacities,
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.inv_viewprojmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                color,
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.settings.to_dict(),
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class SortMode(IntEnum):
    GLOBAL = 0
    PPX_FULL = 1
    PPX_KBUFFER = 2
    HIER = 3
    
    def __str__(self):
        return self.name
    
class GlobalSortOrder(IntEnum):
    Z_DEPTH = 0
    DISTANCE = 1
    PTD_CENTER = 2
    PTD_MAX = 3
    
    def __str__(self):
        return self.name
    
@dataclass
class SortQueueSizes:
    tile_4x4 : int = 64
    tile_2x2 : int = 8
    per_pixel : int = 4
    
    def set_value(self, key, value):
        if key in self.__dataclass_fields__.keys():
            self.__setattr__(key, value)

@dataclass
class SortSettings:
    queue_sizes : SortQueueSizes = field(default_factory=SortQueueSizes)
    sort_mode : SortMode = SortMode.GLOBAL
    sort_order : GlobalSortOrder = GlobalSortOrder.Z_DEPTH
    
    def set_value(self, key, value):
        if key in self.__dataclass_fields__.keys():
            self.__setattr__(key, value)
        else:
            self.queue_sizes.set_value(key, value)

@dataclass
class CullingSettings:
    rect_bounding : bool = False
    tight_opacity_bounding : bool = False
    tile_based_culling : bool = False
    hierarchical_4x4_culling : bool = False
    
    def set_value(self, key, value):
        if key in self.__dataclass_fields__.keys():
            self.__setattr__(key, value)
        
@dataclass
class ExtendedSettings:
    sort_settings : SortSettings = field(default_factory=SortSettings)
    culling_settings : CullingSettings = field(default_factory=CullingSettings)
    load_balancing : bool = False
    proper_ewa_scaling : bool = False
    def to_dict(self):
        return asdict(self, dict_factory=enum_dict_factory)
    def to_json(self):
        return json.dumps(self.to_dict())
    def from_dict(dict):
        return dacite.from_dict(data_class=ExtendedSettings, data=dict, config=dacite.Config(cast=[IntEnum]))
    def from_json(json_filename):
        return ExtendedSettings.from_dict(json.load(open(json_filename)))
    
    def set_value(self, key, value):
        if key in self.__dataclass_fields__.keys():
            self.__setattr__(key, value)
        else:
            self.culling_settings.set_value(key, value)
            self.sort_settings.set_value(key, value)

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    inv_viewprojmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    settings : ExtendedSettings
    render_depth : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

