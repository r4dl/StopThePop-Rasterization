/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

#pragma once

#include "../auxiliary.h"
#include "stopthepop_common.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename T, size_t S>
__device__ void initArray(T(&arr)[S], T v = 0)
{
#pragma unroll
	for (int i = 0; i < S; ++i)
	{
		arr[i] = v;
	}
}

template<int32_t NUM, typename CG, typename KT, typename VT>
__device__ void mergeSortRegToSmem(CG& cg, KT* keys, VT* values, KT* fin_keys, VT* fin_values, KT key, VT value)
{
	// binary search to find location
	int32_t s0 = 0;
#pragma unroll
	for (int32_t i = NUM / 2; i > 0; i /= 2)
	{
		if (keys[s0 + i] <= key)
			s0 += i;
	}
	// move one ahead
	s0 += 1;
	if (keys[0] > key)
		s0 = 0;
	// how many threads of my group are ahead of me
	s0 += cg.thread_rank();

	// reverse the search
	auto store_key = keys[cg.thread_rank()];
	keys[cg.thread_rank()] = key;
	cg.sync();

	// binary search to find location
	int32_t s1 = 0;
#pragma unroll
	for (int32_t i = NUM / 2; i > 0; i /= 2)
	{
		if (keys[s1 + i] < store_key)
			s1 += i;
	}
	// move one ahead
	s1 += 1;
	if (keys[0] >= store_key)
		s1 = 0;
	// how many threads of my group are ahead of me
	s1 += cg.thread_rank();
	auto store_value = values[cg.thread_rank()];
	cg.sync();

	// write out the new order
	fin_keys[s0] = key;
	fin_keys[s1] = store_key;
	fin_values[s0] = value;
	fin_values[s1] = store_value;
	cg.sync();
}

// only works for low number of threads, (WINDOW + THREADS) must be smaller than 2 ^ (32 / THREADS)
template<int32_t THREADS, int32_t WINDOW, typename CG, typename KT, typename VT, typename AF>
__device__ void mergeSortInto(CG& cg, int32_t rank, KT key, VT value, KT* keys, VT* values, AF&& access_function)
{
	// binary search to find location, bias for end
	int loc = WINDOW;
	if (key < keys[access_function(WINDOW - 1)])
	{
		loc = WINDOW - 1;
#pragma unroll
		for (int32_t i = WINDOW / 2; i > 0; i /= 2)
		{
			if (key < keys[access_function(loc - i)])
				loc -= i;
		}
	}
	loc += rank;

	// combined information for all locations, so we can trivially relocate 
	constexpr uint32_t BITS_PER_INFO = 32 / THREADS;
	uint32_t comb_loc = loc << (BITS_PER_INFO * (THREADS - rank - 1));
#pragma unroll
	for (int i = THREADS / 2; i >= 1; i /= 2)
	{
		comb_loc += cg.shfl_xor(comb_loc, i);
	}

	constexpr uint32_t MASK = (0x1 << BITS_PER_INFO) - 1;

	int first_offset = ((comb_loc >> (BITS_PER_INFO * (THREADS - 1))) & MASK) / THREADS * THREADS;
	int move_offset = 4;
	for (int read_from = WINDOW - THREADS + rank; read_from >= first_offset; read_from -= 4)
	{
		while (move_offset > 0 && (comb_loc & MASK) >= read_from + move_offset)
		{
			--move_offset;
			comb_loc = comb_loc >> BITS_PER_INFO;
		}
		
		int read_access = access_function(read_from);
		KT key_move = keys[read_access];
		VT value_move = values[read_access];
		cg.sync();
		if (move_offset > 0)
		{
			int write_access = access_function(read_from + move_offset);
			keys[write_access] = key_move;
			values[write_access] = value_move;
		}
	}
	cg.sync();
	// write my data
	int write_access = access_function(loc);
	keys[write_access] = key;
	values[write_access] = value;
}

template<int N, typename CG, typename KT>
__device__ int shflRankingLocal(CG& cg, int rank, KT key)
{
	// quick ranking with N-1 shfl
	int count = 0;
#pragma unroll
	for (int i = 1; i < N; ++i)
	{
		int other_rank = (rank + i) % N;
		auto other_key = cg.shfl(key, other_rank);
		if (other_key < key ||
			(other_key == key && other_rank < rank))
		{
			++count;
		}
	}
	return count;
}

template<int N, typename CG, typename KT, typename VT>
__device__ void shflSortLocal2Shared(CG& cg, int rank, KT key, VT val, KT* keys, VT* vals)
{
	// quick ranking with 3 shfl
	int count = shflRankingLocal<N>(cg, rank, key);
	keys[count] = key;
	vals[count] = val;
}

// TODO: can we do a better implementation?
template<uint32_t NUM_VALS, typename CG, typename KT, typename VT>
__device__ void batcherSort(CG& cg, KT* keys, VT* vals)
{
	for (uint32_t size = 2; size <= NUM_VALS; size *= 2)
	{
		uint32_t stride = size / 2;
		uint32_t offset = cg.thread_rank() & (stride - 1);

		{
			cg.sync();
			uint32_t pos = 2 * cg.thread_rank() - (cg.thread_rank() & (stride - 1));
			if (keys[pos + 0] > keys[pos + stride])
			{
				swap(keys[pos + 0], keys[pos + stride]);
				swap(vals[pos + 0], vals[pos + stride]);
			}
			stride /= 2;
		}

		for (; stride > 0; stride /= 2)
		{
			cg.sync();
			uint32_t pos = 2 * cg.thread_rank() - (cg.thread_rank() & (stride - 1));

			if (offset >= stride)
			{
				if (keys[pos - stride] > keys[pos + 0])
				{
					swap(keys[pos - stride], keys[pos + 0]);
					swap(vals[pos - stride], vals[pos + 0]);
				}
			}
		}
	}
}



// 0x1 -> tail
// 0x2 -> mid
// 0x4 -> front
// 0x8 -> blend
// 0x10 -> counters and flow
// 0x20 -> general
// 0x40 -> culling
// 0x100 -> select block
#define DEBUG_HIERARCHICAL 0x0

// MID_WINDOW needs to be pow2+4, minimum 8
template <int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA, typename PF, typename SF, typename BF, typename FF>
__device__ void sortGaussiansRayHierarchicaEvaluation(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float4* __restrict__ conic_opacity,
	DebugVisualization debugType,
	PF && prep_function,
	SF && store_function,
	BF && blend_function,
	FF && fin_function)
{
#if (DEBUG_HIERARCHICAL & 0x100) != 0
	//if (blockIdx.x != 7 || blockIdx.y != 7)
	//	return;
	constexpr uint2 target = { 426, 55 };
	if (blockIdx.x != target.x / 16 || blockIdx.y != target.y / 16)
		return;
	uint2 rem = { target.x % 16, target.y % 16 };
	if (cg::this_thread_block().thread_rank() / 32 != rem.x / 8 + (rem.y / 4) * 2)
		return;

	//if (blockIdx.x != 0 || blockIdx.y != 3)
	//	return;
	//if (threadIdx.z != 0)
	//	return;

#endif

	// block size must be: 16,4,4

	// we use the following thread setup per warp
	// 00 01 04 05 16 17 20 21
	// 02 03 06 07 18 19 22 23
	// 08 09 12 13 24 25 28 29
	// 10 11 14 15 26 27 30 31

	// and the following warp setup (which does not matter)
	// 00 01 
	// 02 03 
	// 04 05
	// 06 07

	// and the following half warp setup 
	// 00 01 02 03 
	// 04 05 06 07
	// 08 09 10 11
	// 12 13 14 15

	// every half warp (4x4) block has one smem sort window (32 elements sorted + 32 elements buffer for loading)
	// every 2x2 block has its own 8 element buffer window for local resorting
	// every thread has its own sorted head list typically 4 elements

	// block.thread_index().y/z  identifies the 4x4 tile

	constexpr int PerThreadSortWindow = HEAD_WINDOW;
	constexpr int MidSortWindow = MID_WINDOW;

	// head sorting setup
	float head_depths[PerThreadSortWindow];

	[[maybe_unused]] const uint2 _t1{0, 0};
	decltype(store_function(_t1, 0, 0.0f, 0.0f, 0.0f)) head_stores[PerThreadSortWindow];
	int head_ids[PerThreadSortWindow];

	// mid sorting setup
	__shared__ float mid_depths[4][4][4][MidSortWindow];
	__shared__ int mid_ids[4][4][4][MidSortWindow];
	[[maybe_unused]] uint32_t mid_front = 0;
	[[maybe_unused]] auto mid_access = [&](uint32_t offset)
		{
			return (mid_front + offset) % MidSortWindow;
		};

	// tail sorting setup
	__shared__ float tail_depths[4][4][64];
	__shared__ int tail_ids[4][4][64];

	// tail viewdir is 0, mid viewdirs are 1-4
	__shared__ float3 tail_and_mid_viewdir[4][4][5];

	// global helper
	__shared__ uint2 range;

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<WARP_SIZE>(block);
	auto halfwarp = cg::tiled_partition<WARP_SIZE / 2>(block);
	auto head_group = cg::tiled_partition<4>(halfwarp);


	// initialize head structure
	initArray(head_depths, FLT_MAX);
	initArray(head_stores);
	initArray(head_ids, -1);

	uint32_t fill_counters = 0; // HEAD 8 bit, MID 8 bit, TAIL 16 bit, 
	[[maybe_unused]] constexpr uint32_t FillHeadMask = 0xFF000000;
	constexpr uint32_t FillHeadOne = 0x1000000;
	constexpr uint32_t FillMidMask = 0xFF0000;
	constexpr uint32_t FillMidOne = 0x10000;
	constexpr uint32_t FillTailMask = 0xFFFF;
	constexpr uint32_t FillTailOne = 0x1;

	// initialize ray directions
	const uint2 tile_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 tail_corner = { tile_min.x + 4 * block.thread_index().y, tile_min.y + 4 * block.thread_index().z };

	const glm::mat4 inverse_vp = loadMatrix4x4(projmatrix_inv);
	const float3 campos = *cam_pos;

	if (block.thread_index().x < 5)
	{
		// first thread computes the tail view dir, next 4 the mid view dir
		float2 pos = { static_cast<float>(tail_corner.x), static_cast<float>(tail_corner.y) };
		if (block.thread_index().x == 0)
		{
			pos.x += 1.5f;
			pos.y += 1.5f;
		}
		else
		{
			pos.x += 0.5f + 2 * ((block.thread_index().x - 1) % 2);
			pos.y += 0.5f + 2 * ((block.thread_index().x - 1) / 2);
		}
		float3 dir = computeViewRay(inverse_vp, campos, pos, W, H);
		tail_and_mid_viewdir[block.thread_index().y][block.thread_index().z][block.thread_index().x] = dir;
#if (DEBUG_HIERARCHICAL & 0x2F) != 0 && (DEBUG_HIERARCHICAL & 0x100) != 0
		printf("group dir %d - %d %d %d  - pix %f %f dir %f %f %f\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
			pos.x, pos.y, dir.x, dir.y, dir.z);
#endif
	}

	const int midid = halfwarp.thread_rank() / 4;
	const int midrank = halfwarp.thread_rank() % 4;
	const int midy = midid / 2;
	const int midx = midid % 2;

	const int heady = midrank / 2;
	const int headx = midrank % 2;

	const uint2 pixpos = { tail_corner.x + midx * 2 + headx, tail_corner.y + midy * 2 + heady };
	bool active = pixpos.x < W && pixpos.y < H;

	const float3 viewdir = computeViewRay(inverse_vp, campos, float2{ static_cast<float>(pixpos.x), static_cast<float>(pixpos.y) }, W, H);
#if (DEBUG_HIERARCHICAL & 0x2F) != 0 && (DEBUG_HIERARCHICAL & 0x100) != 0
	printf("own dir %d - %d %d %d  - pix %d %d dir %f %f %f\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
		pixpos.x, pixpos.y, viewdir.x, viewdir.y, viewdir.z);
#endif
	// setup helpers
	const int32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	if (warp.thread_rank() == 0)
	{
		range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	}
	

	if constexpr (MidSortWindow != 8)
	{
		for (int i = 0; i < MidSortWindow; i += 4)
		{
			mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][i + warp.thread_rank() % 4] = FLT_MAX;
			mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][i + warp.thread_rank() % 4] = -1;
		}
	}

	// ensure all helpers are visible
	warp.sync();


	// thread state variables
	auto blend_data = prep_function(active, pixpos);
	
	// lambdas controlling the behavior
	auto blend_one = [&]()
		{
			fill_counters -= FillHeadOne;

			if (!active)
				return;

			if (!blend_function(pixpos, blend_data, head_ids[0], head_stores[0], head_depths[0], debugType))
			{
				active = false;
				return;
			}

#if (DEBUG_HIERARCHICAL & 0x8) != 0
#if (DEBUG_HIERARCHICAL & 0x100)
			if (pixpos.x == target.x && pixpos.y == target.y)
#endif
				printf("%d - %d %d - blending: %f %d %f (%d %d %d)\n", warp.thread_rank(), pixpos.x, pixpos.y,
					head_depths[0], head_ids[0], head_stores[0],
					(fill_counters & FillHeadMask) / FillHeadOne,
					(fill_counters & FillMidMask) / FillMidOne,
					(fill_counters & FillTailMask) / FillTailOne);
#endif

			for (int i = 1; i < PerThreadSortWindow; ++i)
			{
				head_depths[i - 1] = head_depths[i];
				head_stores[i - 1] = head_stores[i];
				head_ids[i - 1] = head_ids[i];
			}
			head_depths[PerThreadSortWindow - 1] = FLT_MAX;
		};



	auto front4OneFromMid = [&](bool checkvalid)
		{
			if (head_group.any(active))
			{
				// prepare depth and data for shfl
				float3 mid_depth_info[3];
				float4 mid_conic_opacity;
				float2 mid_point_xy;

				int load_id;
				if constexpr (MidSortWindow == 8)
				{
					load_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];
				}
				else
				{
					load_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][mid_access(warp.thread_rank() % 4)];
				}
				if (!checkvalid || load_id != -1)
				{
					mid_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
					mid_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
					mid_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
					mid_conic_opacity = conic_opacity[load_id];
					mid_point_xy = points_xy_image[load_id];
				}

#if (DEBUG_HIERARCHICAL & 0x4) != 0
				printf("%d - %d %d - head loading %d\n", warp.thread_rank(), pixpos.x, pixpos.y, load_id);
#endif
				// inner always takes out up to four elements from mid
				for (int inner = 0; inner < 4; ++inner)
				{
					// the head is left most, so this checks for a full sort window
					if (fill_counters >= FillHeadOne * PerThreadSortWindow)
					{
						blend_one();
					}

					// take one from mid
					int coll_id;
					if constexpr (MidSortWindow == 8)
					{
						coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][inner];
					}
					else
					{
						coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][mid_access(inner)];
					}

					// every thread has the same id so this is safe
					if (checkvalid && coll_id == -1)
						continue;

					float3 a = head_group.shfl(mid_depth_info[0], inner);
					float3 b = head_group.shfl(mid_depth_info[1], inner);
					float3 c = head_group.shfl(mid_depth_info[2], inner);

					float depth = depthAlongRay(a, b, c, viewdir);

					// shfl before continue
					float2 xy = head_group.shfl(mid_point_xy, inner);
					float4 con_o = head_group.shfl(mid_conic_opacity, inner);

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - %d new depth %f\n", warp.thread_rank(), pixpos.x, pixpos.y, coll_id, depth);
#endif
					if (!active || depth < 0.0f)
						continue;

					blend_data.contributor++;

					float2 d = { xy.x - static_cast<float>(pixpos.x), xy.y - static_cast<float>(pixpos.y) };

					float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
					if (power > 0.0f)
						continue;

					float G = exp(power);
					float alpha = min(0.99f, con_o.w * G);

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - %d %f alpha is %f\n", warp.thread_rank(), pixpos.x, pixpos.y, coll_id, depth, alpha);
#endif
					if (alpha < 1.0f / 255.0f)
						continue;


					auto store = store_function(pixpos, coll_id, G, alpha, depth);

					// push alpha and depth into per thread sorted array
#pragma unroll
					for (int s = 0; s < PerThreadSortWindow; ++s)
					{
						if (depth < head_depths[s])
						{
							swap(depth, head_depths[s]);
							swap(coll_id, head_ids[s]);
							swap(store, head_stores[s]);
						}
					}
					fill_counters += FillHeadOne;

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - count: %d - sorted: %f %d - %f %d - %f %d - %f %d\n", warp.thread_rank(), pixpos.x, pixpos.y,
						fill_counters, head_depths[0], head_ids[0], head_depths[1], head_ids[1], head_depths[2], head_ids[2], head_depths[3], head_ids[3]);
#endif
				}
			}
			if constexpr (MidSortWindow != 8)
			{
				mid_front += 4;
			}
			fill_counters -= 4 * FillMidOne;
			halfwarp.sync();
		};

	auto pushPullThroughMid = [&](bool checkvalid)
		{
			// prepare depth for shfl
			float3 tail_depth_info[3];
			int load_id = tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x];
			if (!checkvalid || load_id != -1)
			{
				tail_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
				tail_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
				tail_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
			}
			else
			{
				tail_depth_info[0] = tail_depth_info[1] = tail_depth_info[2] = { 0,0,0 };
			}
#if (DEBUG_HIERARCHICAL & 0x2) != 0
			printf("%d - %d %d - mid loading %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, load_id);
#endif

			// take out 4 x 4 elements from tail and move into mid
			for (int mid = 0; mid < 4; ++mid)
			{
				// the tail is the same for everyone in the half warp
				if (checkvalid && (fill_counters & FillTailMask) == 0)
					break;

				// take 4 from tail to mid
				int tid = 4 * mid + (warp.thread_rank() % 4);
				int coll_id = tail_ids[block.thread_index().y][block.thread_index().z][tid];

				float depth = depthAlongRay(halfwarp.shfl(tail_depth_info[0], tid),
					halfwarp.shfl(tail_depth_info[1], tid),
					halfwarp.shfl(tail_depth_info[2], tid),
					tail_and_mid_viewdir[block.thread_index().y][block.thread_index().z][1 + block.thread_index().x / 4]);

				// note: we can only get invalid during draining here
				if (checkvalid && coll_id == -1)
				{
					depth = FLT_MAX;
				}

#if (DEBUG_HIERARCHICAL & 0x2) != 0
				printf("%d - %d %d %d - mid new depth %d %f (%f)\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, tid, coll_id, depth,
					tail_depths[block.thread_index().y][block.thread_index().z][tid]);
#endif
				if constexpr (MidSortWindow == 8)
				{
					// local sort first into front 4 slots (which are empty for sure)
					shflSortLocal2Shared<4>(head_group, warp.thread_rank() % 4, depth, coll_id,
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4]);
					head_group.sync();

					coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];
					depth = mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					printf("%d - %d %d %d - mid local %d sort %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4, coll_id, depth);
#endif


					// we do not need the exact count as we only got invalid during draining
					fill_counters += 4 * FillMidOne;

					// we are not culling here, so we always have data after the first
					// if ( (fill_counters & FillMidMask) > 4 * FillMidOne)
					if (mid != 0 || ((fill_counters & FillMidMask) > 4 * FillMidOne))
					{
						// sort mid					
						mergeSortRegToSmem<4>(head_group,
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4] + 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4] + 4,
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
							depth, coll_id);
						head_group.sync();
#if (DEBUG_HIERARCHICAL & 0x2) != 0
						printf("%d - %d %d %d - sorted into mid %d:  %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4]);
						printf("%d - %d %d %d - sorted into mid %d:  %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, 4 + warp.thread_rank() % 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4]);
#endif
						front4OneFromMid(false);
					}
					else
					{
						// move mid
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4] = coll_id;
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4] = depth;
						// no need to sync here  as shfl of the next iteration will take care of it
					}
				}
				else
				{
					// local sort first using shfl
					int offset = shflRankingLocal<4>(head_group, warp.thread_rank() % 4, depth);
					uint32_t sort_mid_offset = mid_access(MidSortWindow - 4 + offset);
					uint32_t my_mid_offset = mid_access(MidSortWindow - 4 + warp.thread_rank() % 4);
					mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][sort_mid_offset] = coll_id;
					mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][sort_mid_offset] = depth;

					head_group.sync();

					coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][my_mid_offset];
					depth = mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][my_mid_offset];

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					printf("%d - %d %d %d - mid local %d sort %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4, coll_id, depth);
#endif
					// we do not need the exact count as we only got invalid during draining
					fill_counters += 4 * FillMidOne;

					// merge sort with existing
					mergeSortInto<4, MidSortWindow - 4>(head_group, warp.thread_rank() % 4, depth, coll_id,
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_access);

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					for (int j = 0; j < MidSortWindow; j += 4)
					{
						int access = mid_access(j + warp.thread_rank() % 4);
						printf("%d - %d %d %d - sorted into mid %d (%d from %d):  %d %f \n", 
							warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, 
							j + warp.thread_rank() % 4, access, mid_front,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][access],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][access]);
					}
#endif

					// run front if we are full
					if ((fill_counters & FillMidMask) > (MidSortWindow - 4) * FillMidOne)
					{
						front4OneFromMid(false);
					}
				}

				if (checkvalid)
				{
					fill_counters -= min(4 * FillTailOne, FillTailMask & fill_counters);
				}
			}
			if (!checkvalid)
			{
				fill_counters -= 16 * FillTailOne;
			}

		};

	// run through elements and continue to push in and blend out
	for (int progress = range.x; progress < range.y; progress += WARP_SIZE)
	{
		if (!warp.any(active))
			break;

#if DEBUG_HIERARCHICAL != 0
		//if (progress - range.x > 64)
		//	return;
#endif

		// fill new data into tail (last 32 elements) for both tail lists
		// and determine actual elements added and adjust count
		float4 in_conic_opacity;
		float2 in_point_xy;

		int load_id = -1;
		const int tid = progress + warp.thread_rank();
		if (tid < range.y)
		{
			load_id = point_list[tid];
		}

		if (load_id != -1 && CULL_ALPHA)
		{
			in_conic_opacity = conic_opacity[load_id];
			in_point_xy = points_xy_image[load_id];
		}

#if (DEBUG_HIERARCHICAL & 0x1) != 0
		printf("%d - %d %d - loading %d\n", warp.thread_rank(), pixpos.x, pixpos.y, load_id);
#endif

		uint32_t halfs_culled_mask = 0U;
		for (int half = 0; half < 2; ++half)
		{
			int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
			if (load_id != -1)
			{
				// cull against tail tile
				if (CULL_ALPHA)
				{
					// tile boundaries
					const glm::vec2 tail_rect_min = { static_cast<float>(block.group_index().x * BLOCK_X + 4 * xid), static_cast<float>(block.group_index().y * BLOCK_Y + 4 * block.thread_index().z) };
					const glm::vec2 tail_rect_max = { tail_rect_min.x + 3.0f, tail_rect_min.y + 3.0f };

					glm::vec2 max_pos;
					float power = max_contrib_power_rect_gaussian_float<3, 3>(in_conic_opacity, in_point_xy, tail_rect_min, tail_rect_max, max_pos);

					float alpha = min(0.99f, in_conic_opacity.w * exp(-power));
					if (alpha < 1.0f / 255.0f)
						halfs_culled_mask |= (0x1U << half);
				}
			}
		}

		float3 in_depth_info[3];
		if (load_id != -1 && (!CULL_ALPHA || !(halfs_culled_mask == 0x3))) // if culling and not both halfs culled
		{
			in_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
			in_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
			in_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
		}	

		for (int half = 0; half < 2; ++half)
		{
			int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
			float depth = FLT_MAX;

			if (load_id != -1)
			{
				// if not culled, compute depth
				if (!CULL_ALPHA || !(halfs_culled_mask & (0x1U << half)))
				{
					depth = depthAlongRay(in_depth_info[0], in_depth_info[1], in_depth_info[2], tail_and_mid_viewdir[xid][block.thread_index().z][0]);
				}
			}
			
			tail_depths[xid][block.thread_index().z][32 + warp.thread_rank()] = depth;
			tail_ids[xid][block.thread_index().z][32 + warp.thread_rank()] = depth == FLT_MAX ? -1 : load_id;
#if (DEBUG_HIERARCHICAL & 0x1) != 0
			printf("(%d) %d - %d %d %d - %d : %f\n", half, warp.thread_rank(), xid, block.thread_index().z, 0, load_id, depth);
#endif
		}
		// local sort the 32 elements with half warp from shared memory
		batcherSort<32>(halfwarp, tail_depths[block.thread_index().y][block.thread_index().z] + 32, tail_ids[block.thread_index().y][block.thread_index().z] + 32);
		// sync comes through shfl below

#if (DEBUG_HIERARCHICAL & 0x1) != 0
		printf("batcher sort %d/%d: %f %d\n", block.thread_index().y, halfwarp.thread_rank(), tail_depths[block.thread_index().y][block.thread_index().z][32 + halfwarp.thread_rank()], tail_ids[block.thread_index().y][block.thread_index().z][32 + halfwarp.thread_rank()]);
		printf("batcher sort %d/%d: %f %d\n", block.thread_index().y, 16 + halfwarp.thread_rank(), tail_depths[block.thread_index().y][block.thread_index().z][32 + 16 + halfwarp.thread_rank()], tail_ids[block.thread_index().y][block.thread_index().z][32 + 16 + halfwarp.thread_rank()]);
#endif

		for (int half = 0; half < 2; ++half)
		{
			// merge sort if we have old data
			if ((warp.shfl(fill_counters, half * 16) & FillTailMask) != 0)
			{
				int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);

				float* d = tail_depths[xid][block.thread_index().z];
				int* id = tail_ids[xid][block.thread_index().z];

				float k = d[32 + warp.thread_rank()];
				int v = id[32 + warp.thread_rank()];
				// determine number of valid
				uint32_t count_valid = __popc(warp.ballot(v != -1));
				if (half == warp.thread_rank() / 16)
					fill_counters += count_valid * FillTailOne;
				mergeSortRegToSmem<32>(warp, d, id, d, id, k, v);

#if (DEBUG_HIERARCHICAL & 0x1) != 0
				warp.sync();
				printf("merge of %d (%d) sort %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, warp.thread_rank(), d[warp.thread_rank()], id[warp.thread_rank()]);
				printf("merge of %d (%d) sort %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, 32 + warp.thread_rank(), d[32 + warp.thread_rank()], id[32 + warp.thread_rank()]);
#endif
			}
			else
			{
				// copy data to the front
				int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
				float* d = tail_depths[xid][block.thread_index().z];
				int* id = tail_ids[xid][block.thread_index().z];
				d[warp.thread_rank()] = d[32 + warp.thread_rank()];
				int v = id[32 + warp.thread_rank()];
				id[warp.thread_rank()] = v;
				// determine number of valid
				uint32_t count_valid = __popc(warp.ballot(v != -1));
				if (half == warp.thread_rank() / 16)
					fill_counters += count_valid * FillTailOne;

#if (DEBUG_HIERARCHICAL & 0x1) != 0
				warp.sync();
				printf("copied of %d (%d) data %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, warp.thread_rank(), d[warp.thread_rank()], id[warp.thread_rank()]);
#endif
			}
		}

		for (int half = 0; half < 2; ++half)
		{
			if ((fill_counters & FillTailMask) > 32 * FillTailOne)
			{
				// take 16 elements out from mid
				pushPullThroughMid(false);
				halfwarp.sync();

				// move current data in tail (max 48)
				for (int i = 0; i < 3 - half; ++i)
				{
					tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + i * 16] =
						tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + (i + 1) * 16];
					tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + i * 16] =
						tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + (i + 1) * 16];
				}
				halfwarp.sync();
			}

		}
	}

	// debug
#if (DEBUG_HIERARCHICAL & 0x10) != 0
	printf("%d - %d %d %d - draining tail with %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
		(fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif

	if (warp.any(active))
	{
		if ((fill_counters & FillTailMask) != 0)
		{
			for (int half = 0; half < 2; ++half)
			{
				pushPullThroughMid(true);
#if (DEBUG_HIERARCHICAL & 0x10) != 0
				printf("%d - %d %d %d - pulled from mid %d  with %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
					half, (fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif
				if ((half == 0) && (fill_counters & FillTailMask) == 0)
					break;


				// move current data in tail (max 16)
				if (half == 0)
				{
					tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x] =
						tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + 16];
					tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x] =
						tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + 16];
				}
			}
		}

		// drain the remainder from mid
		if (warp.any(active))
		{
			if constexpr (MidSortWindow == 8)
			{
				if ((fill_counters & FillMidMask) != 0)
				{
					// mid still has data, but it is not at the right location, so move it
					mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4] =
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4];
					mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4] =
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4];


					front4OneFromMid(true);

#if (DEBUG_HIERARCHICAL & 0x10) != 0
					printf("%d - %d %d %d - pulled took 4 from mid %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
						(fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif

				}
			}
			else
			{
#if (DEBUG_HIERARCHICAL & 0x10) != 0
				int deb_counter = 0;
#endif
				while ((fill_counters & FillMidMask) != 0)
				{
					front4OneFromMid(true);
#if (DEBUG_HIERARCHICAL & 0x10) != 0
					printf("%d - %d %d %d - pulled (%d) took 4 from mid %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
						deb_counter, (fill_counters& FillHeadMask) / FillHeadOne, (fill_counters& FillMidMask) / FillMidOne, (fill_counters& FillTailMask) / FillTailOne);
					++deb_counter;
#endif
				}
			}
			// drain front
			while (active && fill_counters != 0)
			{
				blend_one();
			}
	}
}


	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.

	if (pixpos.x < W && pixpos.y < H)
	{
		fin_function(pixpos, blend_data, debugType, range.y - range.x);
	}
}



template <int32_t CHANNELS, int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA = true, bool ENABLE_DEBUG_VIZ = false>
__global__ void __launch_bounds__(16 * 16) sortGaussiansRayHierarchicalCUDA_forward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugType,
	float* __restrict__ out_color)
{
	// int num_blends = 0;
	struct BlendDataRaw
	{
		float T;
		float C[CHANNELS];
		uint32_t contributor = 0;
	};
	struct BlendDataWithDebug : public BlendDataRaw
	{
		float errorCounter{};
		float currentDepth;
		float accumDepth;
	};

	using BlendData = std::conditional_t<ENABLE_DEBUG_VIZ, BlendDataWithDebug, BlendDataRaw>;

	auto prep_function = [&](bool inside, const uint2&)
		{
			BlendData bd;
			bd.T = 1.0f;
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				bd.C[ch] = 0.0f;
			}
			if constexpr (ENABLE_DEBUG_VIZ)
			{
				bd.errorCounter = 0.f;
				bd.currentDepth = -FLT_MAX;
				bd.accumDepth = 0.f;
			}
			return bd;
		};
	auto store_function = [](const uint2&, int coll_id, float G, float alpha, float depth)
		{
			return alpha;
		};
	auto blend_function = [&](const uint2& pixpos, BlendData& blend_data, int id, float alpha, float depth, DebugVisualization debugType)
		{
			float test_T = blend_data.T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				return false;
			}

			// TODO: consider using vectors and better loads?
			for (int ch = 0; ch < CHANNELS; ch++)
				blend_data.C[ch] += features[id * CHANNELS + ch] * alpha * blend_data.T;

			// ++num_blends;
			if constexpr (ENABLE_DEBUG_VIZ)
			{
				accumSortingErrorDepth(debugType, blend_data.currentDepth, depth, alpha, blend_data.T, blend_data.accumDepth, blend_data.errorCounter);
			}

			blend_data.T = test_T;

			return true;
		};
	auto fin_function = [&](const uint2& pixpos, BlendData& blend_data, DebugVisualization debugType, int range)
		{
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			final_T[pix_id] = blend_data.T;

			// n_contrib[pix_id] = num_blends;

			if constexpr (!ENABLE_DEBUG_VIZ)
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					out_color[ch * H * W + pix_id] = blend_data.C[ch] + blend_data.T * bg_color[ch];
			}
			else
			{
				outputDebugVis(debugType, out_color, pix_id, blend_data.contributor, blend_data.T, blend_data.accumDepth, blend_data.errorCounter, range, H, W);
			}
		};

	sortGaussiansRayHierarchicaEvaluation<HEAD_WINDOW, MID_WINDOW, CULL_ALPHA>(
		ranges, point_list, W, H, points_xy_image, cov3Ds_inv, projmatrix_inv, cam_pos, conic_opacity, debugType,
		prep_function, store_function, blend_function, fin_function);
}


template <int32_t CHANNELS, int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA = true>
__global__ void __launch_bounds__(16 * 16) sortGaussiansRayHierarchicalCUDA_backward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;
	// int num_blends = 0;
	struct BlendData
	{
		float T_final;
		float dL_dpixel[CHANNELS];
		float final_color[CHANNELS];
		float T;
		float C[CHANNELS];
		uint32_t contributor = 0;
	};
	auto prep_function = [&](bool inside, const uint2& pixpos)
		{
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			BlendData bd;
			bd.T = 1.0f;
			bd.T_final = inside ? final_Ts[pix_id] : 0;
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				bd.C[ch] = 0.0f;
				if (inside)
				{
					bd.dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
					bd.final_color[ch] = pixel_colors[ch * H * W + pix_id] - bd.T_final * bg_color[ch];
				}

			}
				
			return bd;
		};
	auto store_function = [](const uint2&, int coll_id, float G, float alpha, float depth)
		{
			return G;
		};
	auto blend_function = [&](const uint2& pixpos, BlendData& blend_data, int global_id, float G, float depth, DebugVisualization debugType)
		{
			const float4 con_o = conic_opacity[global_id];

			const float alpha = min(0.99f, con_o.w * G);
			float test_T = blend_data.T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				return false;
			}

			// ++num_blends;

			const float2 xy = points_xy_image[global_id];
			const float2 d = { xy.x - static_cast<float>(pixpos.x), xy.y - static_cast<float>(pixpos.y) };


			const float dchannel_dcolor = alpha * blend_data.T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				const float c = colors[global_id * CHANNELS + ch];

				// reconstruct color up to this point
				blend_data.C[ch] += c * alpha * blend_data.T;
				// the contribution of all other gaussian coming after
				float accum_rec_ch = (blend_data.final_color[ch] - blend_data.C[ch]) / test_T;

				const float dL_dchannel = blend_data.dL_dpixel[ch];
				dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * CHANNELS + ch]), dchannel_dcolor * dL_dchannel);

			}
			dL_dalpha *= blend_data.T;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < CHANNELS; i++)
				bg_dot_dpixel += bg_color[i] * blend_data.dL_dpixel[i];
			dL_dalpha += (-blend_data.T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			blend_data.T = test_T;

			return true;
		};
	auto fin_function = [&](const uint2& pixpos, BlendData& blend_data, DebugVisualization debugType, int range)
		{
			return;
		};

	sortGaussiansRayHierarchicaEvaluation<HEAD_WINDOW, MID_WINDOW, CULL_ALPHA>(
		ranges, point_list, W, H, points_xy_image, cov3Ds_inv, projmatrix_inv, cam_pos, conic_opacity, DebugVisualization::Disabled,
		prep_function, store_function, blend_function, fin_function);
}
