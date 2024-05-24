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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "stopthepop/stopthepop_common.cuh"
#include "stopthepop/rasterizer_debug.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

void applyDebugVisualization(
	DebugVisualizationData& debugVisualization,
	int width, int height,
	CudaRasterizer::ImageState& imgState,
	CudaRasterizer::BinningState& binningState,
	const float2* means2D,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float* scales,
	const float* rotations,
	float* out_color,
	bool debug
)
{
	if (sortQualityDebug::isVisualized(debugVisualization.type))
	{
		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		float* d_min_max; // GPU pointer for the result
		cudaMalloc((void**)&d_min_max, sizeof(float) * 2);

		int N = width * height;

		cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, out_color, d_min_max + 1, N);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, out_color, d_min_max, N);
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, out_color, d_min_max + 1, N);

		std::array<float, 2> min_max_contribution_count;
		cudaMemcpy(min_max_contribution_count.data(), d_min_max, 2 * sizeof(float), cudaMemcpyDeviceToHost);

		float value = 0;
		if (debugVisualization.debugPixel[0] > 0 && debugVisualization.debugPixel[0] < width && debugVisualization.debugPixel[1] > 0 && debugVisualization.debugPixel[1] < height)
		{
			uint32_t pix_id = width * debugVisualization.debugPixel[1] + debugVisualization.debugPixel[0];
			cudaMemcpy(&value, out_color + pix_id, sizeof(float), cudaMemcpyDeviceToHost);
		}

		// Statistics
		// Avg, STD
		std::vector<float> data(N);
		cudaMemcpy(data.data(), out_color, sizeof(float) * N, cudaMemcpyDeviceToHost);

		float sum = std::accumulate(data.begin(), data.end(), 0.0f, std::plus<float>());
		float average = sum / static_cast<float>(N);
		float std = std::sqrt(std::accumulate(data.begin(), data.end(), 0.f, [average](float v, float n) {
				return v + ((n - average) * (n - average));
				}) / static_cast<float>(N));

		debugVisualization.dataCallback(debugVisualization, value, min_max_contribution_count[0], min_max_contribution_count[1], average, std);
		CHECK_CUDA(FORWARD::render_debug(debugVisualization, width * height, out_color, d_min_max), debug)
		
		cudaFree(d_min_max);
	}
 }

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	const glm::mat4x3 viewmatrix_mat = loadMatrix4x3(viewmatrix);

	glm::vec3 p_view;
	present[idx] = in_frustum(idx, mean3D, viewmatrix_mat, false, p_view);
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != INVALID_TILE_ID;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			if (valid_tile) 
				ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, bool requires_cov3D_inv)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.rects2D, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	if (requires_cov3D_inv)
		obtain(chunk, geom.cov3D_inv, P * 3, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const SplattingSettings splatting_settings,
	DebugVisualizationData& debugVisualization,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	static Timer timer({ "Preprocess", "Duplicate", "Sort", "Render" });
	timer.setActive(debugVisualization.timing_enabled);

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	bool requires_cov3D_inv = splatting_settings.sort_settings.requiresDepthAlongRay();
	size_t chunk_size = required<GeometryState>(P, requires_cov3D_inv);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, requires_cov3D_inv);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
	timer();

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.rects2D,
		splatting_settings,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.cov3D_inv,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	timer();

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	FORWARD::duplicate(
		P,
		geomState.means2D,
		geomState.conic_opacity,
		radii,
		geomState.rects2D,
		geomState.point_offsets,
		geomState.depths,
		geomState.cov3D_inv,
		splatting_settings,
		projmatrix,
		inv_viewprojmatrix,
		cam_pos,
		width, height,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		tile_grid);
	CHECK_CUDA(, debug)

	timer();

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	timer();

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		splatting_settings,
		binningState.point_list,
		width, height,
		geomState.means2D,
		means3D,
		geomState.cov3D_inv,
		inv_viewprojmatrix,
		(glm::vec3*)cam_pos,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		debugVisualization,
		out_color), debug)

	timer();

	std::vector<std::pair<std::string, float>> timings;
	timer.syncAddReport(timings);

	if (timings.size() > 0)
	{
		std::stringstream ss;
		ss << "Timings: \n";
		for (auto const& x : timings)
			ss << " - " << x.first << ": " << x.second << "ms\n";
		debugVisualization.timings_text = ss.str();
		// std::cout << debugVisualization.timings_text << std::endl;
	}

	applyDebugVisualization(
		debugVisualization,
		width, height,
		imgState, binningState, geomState.means2D,
		viewmatrix, projmatrix, cam_pos,
		scales, rotations,
		out_color,
		debug
	);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const SortSettings sort_settings,
	const CullingSettings culling_settings,
	const bool proper_ewa_scaling,
	const float* means3D,
	const float* shs,
    const float* opacities,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const float* pixel_colors,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	bool requires_cov3D_inv = sort_settings.requiresDepthAlongRay();
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, requires_cov3D_inv);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid, block,
		imgState.ranges,
		sort_settings,
		culling_settings,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.cov3D_inv,
		inv_viewprojmatrix,
		(glm::vec3*)cam_pos,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		pixel_colors,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		proper_ewa_scaling,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)cam_pos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}