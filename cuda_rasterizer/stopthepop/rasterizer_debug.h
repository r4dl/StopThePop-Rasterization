/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

#pragma once

#include <functional>
#include <string>

enum class DebugVisualization
{
	SortErrorOpacity,
	SortErrorDistance,
	GaussianCountPerTile,
	GaussianCountPerPixel,
	Depth,
	Transmittance,
	Disabled,
};


inline std::string toString(DebugVisualization m) {
	switch(m)
	{
		case DebugVisualization::SortErrorOpacity:
			return "Sort Error: Opacity";
		case DebugVisualization::SortErrorDistance:
			return "Sort Error: Distance";
		case DebugVisualization::GaussianCountPerTile:
			return "Gaussian Count Per Tile";
		case DebugVisualization::GaussianCountPerPixel:
			return "Gaussian Count Per Pixel";
		case DebugVisualization::Depth:
			return "Depth";
		case DebugVisualization::Transmittance:
			return "Transmittance";
		default:
			return "Disabled";
	}
};

struct DebugVisualizationData
{
	DebugVisualization type{DebugVisualization::Disabled};
	int debugPixel[2] = {};
	// Parameter: this, value of debugPixelX/Y, min of frame, max of frame, average, std
	std::function<void(const DebugVisualizationData&, float, float, float, float, float)> dataCallback{
		[](const DebugVisualizationData&, float, float, float, float, float) {}
	};
	float minMax[2] = { 0.f, 10000.f};
	bool debug_normalize = false;

	std::string timings_text = "";
	bool timing_enabled = false;
};

namespace sortQualityDebug
{
__device__ __host__ inline bool isSortError(DebugVisualization v)
{
	switch(v)
	{
		case DebugVisualization::SortErrorDistance:
		case DebugVisualization::SortErrorOpacity:
			return true;
		default:
			return false;
	}
}

__device__ __host__ inline bool isMagma(DebugVisualization v)
{
	switch(v)
	{
		case DebugVisualization::SortErrorDistance:
		case DebugVisualization::SortErrorOpacity:
		case DebugVisualization::Transmittance:
		case DebugVisualization::GaussianCountPerPixel:
		case DebugVisualization::GaussianCountPerTile:
			return true;
		default:
			return false;
	}
}

__device__ __host__ inline bool isVisualized(DebugVisualization v)
{
	switch(v)
	{
		case DebugVisualization::SortErrorDistance:
		case DebugVisualization::SortErrorOpacity:
		case DebugVisualization::Transmittance:
		case DebugVisualization::GaussianCountPerPixel:
		case DebugVisualization::GaussianCountPerTile:
		case DebugVisualization::Depth:
			return true;
		default:
			return false;
	}
}

}
