/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "rasterizer.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

using namespace CudaRasterizer;

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* rects,
		const SplattingSettings splatting_settings,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float4* cov3D_invs,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	void duplicate(int P,
		const float2 *means2D,
		const float4 *conic_opacity,
		const int *radii,
		const float2 *rects2D,
		const uint32_t *offsets,
		const float *depths,
		const float4 *cov3D_invs,
		const SplattingSettings splatting_settings,
		const float *projmatrix,
		const float *inv_viewprojmatrix,
		const float *cam_pos,
		const int W, int H,
		uint64_t *gaussian_keys_unsorted,
		uint32_t *gaussian_values_unsorted,
		dim3 grid);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const SplattingSettings splatting_settings,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* means3D,
		const float4* cov3D_inv,
		const float* projmatrix_inv,
		const glm::vec3* cam_pos,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		DebugVisualizationData& debugVisualization,
		float* out_color);

	void render_debug(DebugVisualizationData& debugVisualization, int P, float* out_color, float* min_max_contrib);

}


#endif