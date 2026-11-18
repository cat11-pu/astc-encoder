// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2026 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @file
 * @brief Deterministic property tests for illegal inputs and reference encodings.
 *
 * The illegal corpus contains 200 cases, split into four groups of 50:
 *
 *   * negative image dimensions, represented as sign-extended size_t values
 *   * invalid block geometry and derived block coordinates
 *   * oversized quality and codec tuning parameters
 *   * duplicated ASTC headers alternating with valid and invalid payloads
 *
 * Every illegal case is expected to fail cleanly, leave caller buffers
 * untouched, and free all owned allocations. Run this executable under ASAN,
 * UBSAN, and LSAN to turn memory errors, undefined behavior, and leaks into a
 * nonzero process exit.
 *
 * The legal corpus contains 50 image/settings pairs. The reference build runs
 * with --emit-reference to write .astc files. The instrumented build runs with
 * --verify-reference to re-encode each image and compare every 16-byte block
 * against the reference payload.
 */

#include "astcenc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace
{

constexpr size_t GUARD_SIZE = 64;
constexpr uint8_t GUARD_HEAD = 0xA5;
constexpr uint8_t GUARD_TAIL = 0x5A;
constexpr uint32_t ASTC_MAGIC = 0x5CA1AB13u;

struct GuardedBuffer
{
	std::vector<uint8_t> storage;
	size_t offset;
	size_t length;

	explicit GuardedBuffer(size_t size) :
	    storage(size + 2 * GUARD_SIZE, 0),
	    offset(GUARD_SIZE),
	    length(size)
	{
		std::memset(storage.data(), GUARD_HEAD, GUARD_SIZE);
		std::memset(storage.data() + offset, 0, size);
		std::memset(storage.data() + offset + size, GUARD_TAIL, GUARD_SIZE);
	}

	uint8_t* data()
	{
		return storage.data() + offset;
	}

	const uint8_t* data() const
	{
		return storage.data() + offset;
	}

	size_t size() const
	{
		return length;
	}

	bool intact() const
	{
		for (size_t i = 0; i < GUARD_SIZE; i++)
		{
			if (storage[i] != GUARD_HEAD)
			{
				return false;
			}

			if (storage[offset + length + i] != GUARD_TAIL)
			{
				return false;
			}
		}

		return true;
	}
};

struct CaseResult
{
	int id;
	std::string category;
	std::string name;
	bool pass;
	std::string detail;
};

struct Totals
{
	size_t cases { 0 };
	size_t passed { 0 };
	size_t failed { 0 };
	size_t api_calls { 0 };
	size_t legal_cases { 0 };
	size_t legal_blocks { 0 };
	size_t block_mismatches { 0 };
};

std::vector<CaseResult> g_results;
Totals g_totals;

std::string hex_bytes(const uint8_t* data, size_t size)
{
	static const char* digits = "0123456789abcdef";
	std::string out;
	out.reserve(size * 2);

	for (size_t i = 0; i < size; i++)
	{
		out.push_back(digits[data[i] >> 4]);
		out.push_back(digits[data[i] & 0xF]);
	}

	return out;
}

const char* status_name(astcenc_error status)
{
	return astcenc_get_error_string(status);
}

bool is_clean_rejection(astcenc_error status)
{
	return status == ASTCENC_ERR_BAD_PARAM || status == ASTCENC_ERR_OUT_OF_MEM;
}

bool is_block_decode_status(astcenc_error status)
{
	return status == ASTCENC_SUCCESS || status == ASTCENC_ERR_BAD_BLOCK;
}

void record_case(
	int id,
	const char* category,
	std::string name,
	bool pass,
	std::string detail
) {
	g_results.push_back({ id, category, std::move(name), pass, std::move(detail) });
	g_totals.cases++;

	if (pass)
	{
		g_totals.passed++;
	}
	else
	{
		g_totals.failed++;
	}
}

size_t component_size(astcenc_type type)
{
	switch (type)
	{
	case ASTCENC_TYPE_U8:
		return 1;
	case ASTCENC_TYPE_F16:
		return 2;
	case ASTCENC_TYPE_F32:
		return 4;
	default:
		return 1;
	}
}

size_t div_round_up(size_t value, size_t divisor)
{
	return (value + divisor - 1) / divisor;
}

size_t block_count(size_t dx, size_t dy, size_t dz, size_t bx, size_t by, size_t bz)
{
	return div_round_up(dx, bx) * div_round_up(dy, by) * div_round_up(dz, bz);
}

struct ImageFixture
{
	GuardedBuffer pixels;
	std::vector<void*> slices;
	astcenc_image image {};

	ImageFixture(
	    size_t dx,
	    size_t dy,
	    size_t dz,
	    astcenc_type type,
	    uint64_t seed
	) :
	    pixels(dx * dy * dz * 4 * component_size(type)),
	    slices(std::min<size_t>(dz, 16), pixels.data())
	{
		uint8_t* base = pixels.data();
		size_t slice_bytes = dx * dy * 4 * component_size(type);

		for (size_t z = 0; z < slices.size(); z++)
		{
			slices[z] = base + z * slice_bytes;
		}

		// Fill with a deterministic, high-entropy byte pattern.
		uint8_t* p = base;
		for (size_t i = 0; i < pixels.size(); i++)
		{
			uint64_t v = (seed + 0x9E3779B97F4A7C15ull) * (i + 0xD1B54A32D192ED03ull);
			v ^= v >> 29;
			p[i] = static_cast<uint8_t>(v);
		}

		image.dim_x = dx;
		image.dim_y = dy;
		image.dim_z = dz;
		image.data_type = type;
		image.data = slices.data();
		image.row_stride = 0;
	}
};

struct ContextGuard
{
	astcenc_context* context { nullptr };

	~ContextGuard()
	{
		if (context)
		{
			astcenc_context_free(context);
		}
	}
};

astcenc_error make_context(
	astcenc_profile profile,
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	float quality,
	unsigned int flags,
	ContextGuard& guard,
	astcenc_config* config_out = nullptr
) {
	astcenc_config config {};
	astcenc_error status = astcenc_config_init(
	    profile, block_x, block_y, block_z, quality, flags, &config);
	if (status != ASTCENC_SUCCESS)
	{
		return status;
	}

	if (config_out)
	{
		*config_out = config;
	}

	return astcenc_context_alloc(&config, 1, &guard.context, nullptr);
}

constexpr astcenc_swizzle SWIZZLE_RGBA {
	ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
};

constexpr astcenc_swizzle SWIZZLE_RRRG {
	ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_G
};

//
// Group 1: negative image dimensions.
//

void run_negative_dimension_cases()
{
	static const size_t bad_values[] {
		std::numeric_limits<size_t>::max(),
		std::numeric_limits<size_t>::max() - 1,
		std::numeric_limits<size_t>::max() / 2,
		std::numeric_limits<size_t>::max() / 4,
		static_cast<size_t>(0xFFFFFFFFu),
		static_cast<size_t>(0x80000000u),
		static_cast<size_t>(0x100000000ull),
		static_cast<size_t>(1ull << 40)
	};

	static const std::array<std::array<unsigned int, 3>, 4> blocks {{
		{{ 4, 4, 1 }},
		{{ 6, 6, 1 }},
		{{ 12, 12, 1 }},
		{{ 6, 6, 6 }}
	}};

	for (int i = 0; i < 50; i++)
	{
		size_t dx = 4;
		size_t dy = 4;
		size_t dz = 1;
		size_t row_stride = 0;
		astcenc_type type = (i % 7 == 0) ? ASTCENC_TYPE_F32 :
		                    (i % 5 == 0) ? ASTCENC_TYPE_F16 :
		                                   ASTCENC_TYPE_U8;

		size_t bad = bad_values[i % (sizeof(bad_values) / sizeof(bad_values[0]))];
		switch (i % 6)
		{
		case 0:
			dx = bad;
			break;
		case 1:
			dy = bad;
			break;
		case 2:
			dz = bad;
			break;
		case 3:
			dx = bad;
			dy = 1;
			break;
		case 4:
			dx = bad;
			dz = bad;
			break;
		default:
			dx = 4;
			dy = bad;
			row_stride = std::numeric_limits<size_t>::max();
			break;
		}

		if (i % 11 == 0)
		{
			dx = 0;
		}
		else if (i % 13 == 0)
		{
			dy = 0;
		}
		else if (i % 17 == 0)
		{
			dz = 0;
		}

		const auto& block = blocks[i % blocks.size()];
		ContextGuard context;
		astcenc_error status = make_context(
		    (i % 2) ? ASTCENC_PRF_LDR : ASTCENC_PRF_LDR_SRGB,
		    block[0], block[1], block[2], ASTCENC_PRE_FASTEST, 0, context);

		if (status != ASTCENC_SUCCESS)
		{
			record_case(i + 1, "negative_dimensions",
			            "context setup failed unexpectedly", false,
			            status_name(status));
			continue;
		}

		// Deliberately undersized fixtures. The API must reject the descriptor
		// before touching these buffers.
		ImageFixture fixture(4, 4, 1, type, static_cast<uint64_t>(i + 1));
		fixture.image.dim_x = dx;
		fixture.image.dim_y = dy;
		fixture.image.dim_z = dz;
		fixture.image.row_stride = row_stride;

		GuardedBuffer in(4096);
		GuardedBuffer out(4096);
		bool compress = (i % 2) == 0;

		if (compress)
		{
			status = astcenc_compress_image(
			    context.context, &fixture.image, &SWIZZLE_RGBA,
			    out.data(), out.size(), 0);
		}
		else
		{
			status = astcenc_decompress_image(
			    context.context, in.data(), in.size(), &fixture.image,
			    &SWIZZLE_RGBA, 0);
		}
		g_totals.api_calls++;

		bool pass = is_clean_rejection(status)
		         && fixture.pixels.intact()
		         && in.intact()
		         && out.intact();

		char detail[512];
		std::snprintf(detail, sizeof(detail),
		              "mode=%s dims=%zu,%zu,%zu type=%d row_stride=%zu status=%s guards=%s",
		              compress ? "compress" : "decompress",
		              dx, dy, dz, static_cast<int>(type), row_stride,
		              status_name(status),
		              (fixture.pixels.intact() && in.intact() && out.intact()) ? "ok" : "corrupt");

		record_case(i + 1, "negative_dimensions",
		            std::string(compress ? "compress_" : "decompress_") + std::to_string(i),
		            pass, detail);
	}
}

//
// Group 2: invalid block geometry and derived block coordinates.
//

void run_block_geometry_cases()
{
	struct BadBlock
	{
		unsigned int x;
		unsigned int y;
		unsigned int z;
	};

	static const BadBlock bad_blocks[] {
		{ 0, 0, 1 },
		{ 0, 4, 1 },
		{ 4, 0, 1 },
		{ 4, 5, 1 },
		{ 3, 3, 1 },
		{ 13, 13, 1 },
		{ 1, 1, 1 },
		{ 4, 4, 2 },
		{ 3, 3, 4 },
		{ 7, 7, 7 },
		{ 255, 255, 1 },
		{ std::numeric_limits<unsigned int>::max(),
		  std::numeric_limits<unsigned int>::max(), 1 },
		{ 2, 2, 2 },
		{ 12, 12, 12 }
	};

	for (int i = 0; i < 50; i++)
	{
		bool pass = false;
		std::string name;
		std::string detail;

		if (i < 14)
		{
			const BadBlock& block = bad_blocks[i];
			astcenc_config config {};
			astcenc_error status = astcenc_config_init(
			    ASTCENC_PRF_LDR, block.x, block.y, block.z,
			    ASTCENC_PRE_FASTEST, 0, &config);
			g_totals.api_calls++;

			pass = status == ASTCENC_ERR_BAD_BLOCK_SIZE
			    || status == ASTCENC_ERR_NOT_IMPLEMENTED;
			name = "invalid_block_size_" + std::to_string(i);
			detail = "block=" + std::to_string(block.x) + "x"
			       + std::to_string(block.y) + "x"
			       + std::to_string(block.z) + " status=" + status_name(status);
		}
		else
		{
			ContextGuard context;
			astcenc_error setup = make_context(
			    ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FASTEST, 0, context);
			if (setup != ASTCENC_SUCCESS)
			{
				record_case(51 + i, "bad_block_coordinates",
				            "context setup failed unexpectedly", false,
				            status_name(setup));
				continue;
			}

			ImageFixture fixture(4, 4, 1, ASTCENC_TYPE_U8, static_cast<uint64_t>(100 + i));
			GuardedBuffer in(4096);
			GuardedBuffer out(4096);
			astcenc_error status = ASTCENC_SUCCESS;

			if (i < 22)
			{
				// A 2D block cannot address a volumetric image.
				fixture.image.dim_z = 2 + static_cast<size_t>(i % 4);
				status = astcenc_compress_image(
				    context.context, &fixture.image, &SWIZZLE_RGBA,
				    out.data(), out.size(), 0);
				pass = status == ASTCENC_ERR_BAD_PARAM;
				name = "2d_block_3d_image_" + std::to_string(i);
			}
			else if (i < 30)
			{
				// A row pitch that cannot address one full row.
				// 1/7/15 are below the 16-byte row pitch for a 4x4 U8
				// image. SIZE_MAX makes the slice extent overflow.
				fixture.image.row_stride = (i % 3 == 0)
				    ? 15
				    : (i % 3 == 1) ? 7 : std::numeric_limits<size_t>::max();
				status = astcenc_compress_image(
				    context.context, &fixture.image, &SWIZZLE_RGBA,
				    out.data(), out.size(), 0);
				pass = status == ASTCENC_ERR_BAD_PARAM;
				name = "invalid_row_pitch_" + std::to_string(i);
			}
			else if (i < 38)
			{
				// Dimensions that make derived block coordinates or byte
				// extents overflow.
				size_t axis = static_cast<size_t>(i % 3);
				size_t value = std::numeric_limits<size_t>::max() / 8 + 1;
				if (axis == 0)
				{
					fixture.image.dim_x = value;
				}
				else if (axis == 1)
				{
					fixture.image.dim_y = value;
				}
				else
				{
					fixture.image.dim_z = value;
				}
				status = (i % 2)
				    ? astcenc_compress_image(context.context, &fixture.image,
				                             &SWIZZLE_RGBA, out.data(), out.size(), 0)
				    : astcenc_decompress_image(context.context, in.data(), in.size(),
				                               &fixture.image, &SWIZZLE_RGBA, 0);
				pass = is_clean_rejection(status);
				name = "block_coordinate_overflow_" + std::to_string(i);
			}
			else if (i < 44)
			{
				// The block map requires more payload than supplied.
				fixture.image.dim_x = 8;
				fixture.image.dim_y = 8;
				size_t data_len = (i % 2) ? 63 : 0;
				status = astcenc_decompress_image(
				    context.context, in.data(), data_len, &fixture.image,
				    &SWIZZLE_RGBA, 0);
				pass = status == ASTCENC_ERR_OUT_OF_MEM;
				name = "short_block_map_" + std::to_string(i);
			}
			else if (i < 47)
			{
				status = astcenc_compress_image(
				    context.context, &fixture.image, &SWIZZLE_RGBA,
				    out.data(), out.size(), 1);
				pass = status == ASTCENC_ERR_BAD_PARAM;
				name = "invalid_thread_coordinate_" + std::to_string(i);
			}
			else
			{
				// F16 rows must be a multiple of two bytes in size. A
				// 33-byte pitch is large enough but misaligned.
				fixture.image.data_type = ASTCENC_TYPE_F16;
				fixture.image.row_stride = 33;
				status = astcenc_compress_image(
				    context.context, &fixture.image, &SWIZZLE_RGBA,
				    out.data(), out.size(), 0);
				pass = status == ASTCENC_ERR_BAD_PARAM;
				name = "misaligned_f16_row_pitch_" + std::to_string(i);
			}
			g_totals.api_calls++;

			pass = pass && fixture.pixels.intact() && in.intact() && out.intact()
			    && std::strcmp(status_name(status), "ASTCENC_SUCCESS") != 0;
			detail = "status=" + std::string(status_name(status));
		}

		record_case(51 + i, "bad_block_coordinates", name, pass, detail);
	}
}

//
// Group 3: oversized quality and codec tuning parameters.
//

void run_oversized_quality_cases()
{
	static const float bad_quality[] {
		100.0001f,
		101.0f,
		1.0e6f,
		std::numeric_limits<float>::max(),
		std::numeric_limits<float>::infinity(),
		-std::numeric_limits<float>::infinity(),
		std::numeric_limits<float>::quiet_NaN(),
		-1.0f
	};

	static const std::array<std::array<unsigned int, 3>, 4> blocks {{
		{{ 4, 4, 1 }},
		{{ 6, 5, 1 }},
		{{ 12, 12, 1 }},
		{{ 6, 6, 6 }}
	}};

	for (int i = 0; i < 50; i++)
	{
		bool pass = false;
		std::string name;
		std::string detail;

		if (i < 24)
		{
			float quality = bad_quality[i % (sizeof(bad_quality) / sizeof(bad_quality[0]))];
			const auto& block = blocks[i % blocks.size()];
			astcenc_config config {};
			astcenc_error status = astcenc_config_init(
			    static_cast<astcenc_profile>(i % 4), block[0], block[1], block[2],
			    quality, (i % 3 == 0) ? ASTCENC_FLG_USE_DECODE_UNORM8 : 0,
			    &config);
			g_totals.api_calls++;

			pass = status == ASTCENC_ERR_BAD_QUALITY;
			name = "invalid_quality_" + std::to_string(i);
			detail = "quality=" + std::to_string(quality)
			       + " status=" + status_name(status);
		}
		else
		{
			astcenc_config config {};
			astcenc_error status = astcenc_config_init(
			    ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FASTEST, 0, &config);
			if (status != ASTCENC_SUCCESS)
			{
				record_case(101 + i, "oversized_quality_parameters",
				            "default config setup failed unexpectedly", false,
				            status_name(status));
				continue;
			}

			unsigned int variant = static_cast<unsigned int>(i - 24);
			switch (variant % 13)
			{
			case 0:
				config.a_scale_radius = std::numeric_limits<unsigned int>::max();
				break;
			case 1:
				config.rgbm_m_scale = std::numeric_limits<float>::max();
				break;
			case 2:
				config.cw_r_weight = std::numeric_limits<float>::max();
				config.cw_g_weight = std::numeric_limits<float>::max();
				config.cw_b_weight = std::numeric_limits<float>::max();
				config.cw_a_weight = std::numeric_limits<float>::max();
				break;
			case 3:
				config.tune_db_limit = std::numeric_limits<float>::max();
				break;
			case 4:
				config.tune_mse_overshoot = std::numeric_limits<float>::max();
				break;
			case 5:
				config.tune_2plane_early_out_limit_correlation =
				    std::numeric_limits<float>::max();
				break;
			case 6:
				config.tune_block_mode_limit = std::numeric_limits<unsigned int>::max();
				break;
			case 7:
				config.tune_candidate_limit = std::numeric_limits<unsigned int>::max();
				break;
			case 8:
				config.tune_partition_count_limit = std::numeric_limits<unsigned int>::max();
				break;
			case 9:
				config.tune_2partition_index_limit = std::numeric_limits<unsigned int>::max();
				break;
			case 10:
				config.tune_3partitioning_candidate_limit =
				    std::numeric_limits<unsigned int>::max();
				break;
			case 11:
				config.tune_4partitioning_candidate_limit =
				    std::numeric_limits<unsigned int>::max();
				break;
			default:
				config.tune_2partition_early_out_limit_factor =
				    std::numeric_limits<float>::max();
				break;
			}

			ContextGuard context;
			status = astcenc_context_alloc(&config, 1, &context.context, nullptr);
			if (status != ASTCENC_SUCCESS)
			{
				record_case(101 + i, "oversized_quality_parameters",
				            "oversized tuning param context alloc", false,
				            status_name(status));
				continue;
			}

			ImageFixture fixture(8, 8, 1, ASTCENC_TYPE_U8, static_cast<uint64_t>(200 + i));
			GuardedBuffer out(4 * 16);
			status = astcenc_compress_image(
			    context.context, &fixture.image, &SWIZZLE_RGBA,
			    out.data(), out.size(), 0);
			g_totals.api_calls++;

			pass = status == ASTCENC_SUCCESS
			    && fixture.pixels.intact()
			    && out.intact();
			name = "oversized_tuning_" + std::to_string(variant);
			detail = "variant=" + std::to_string(variant % 13)
			       + " status=" + status_name(status)
			       + " guards=" + (fixture.pixels.intact() && out.intact() ? "ok" : "corrupt");
		}

		record_case(101 + i, "oversized_quality_parameters", name, pass, detail);
	}
}

//
// Group 4: duplicate ASTC headers alternating with payload blocks.
//

void append_header(std::vector<uint8_t>& out, unsigned int block_x, unsigned int block_y,
                   unsigned int block_z, size_t dim_x, size_t dim_y, size_t dim_z,
                   uint32_t magic)
{
	auto push_u24 = [&out](size_t value) {
		out.push_back(static_cast<uint8_t>(value));
		out.push_back(static_cast<uint8_t>(value >> 8));
		out.push_back(static_cast<uint8_t>(value >> 16));
	};

	out.push_back(static_cast<uint8_t>(magic));
	out.push_back(static_cast<uint8_t>(magic >> 8));
	out.push_back(static_cast<uint8_t>(magic >> 16));
	out.push_back(static_cast<uint8_t>(magic >> 24));
	out.push_back(static_cast<uint8_t>(block_x));
	out.push_back(static_cast<uint8_t>(block_y));
	out.push_back(static_cast<uint8_t>(block_z));
	push_u24(dim_x);
	push_u24(dim_y);
	push_u24(dim_z);
}

void run_duplicate_header_cases()
{
	// Produce one known-valid 4x4 block for interleaving.
	uint8_t valid_block[16] {};
	{
		ContextGuard context;
		astcenc_error status = make_context(
		    ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FASTEST, 0, context);
		if (status == ASTCENC_SUCCESS)
		{
			ImageFixture fixture(4, 4, 1, ASTCENC_TYPE_U8, 0x12345678u);
			status = astcenc_compress_image(
			    context.context, &fixture.image, &SWIZZLE_RGBA,
			    valid_block, sizeof(valid_block), 0);
		}

		if (status != ASTCENC_SUCCESS)
		{
			record_case(151, "duplicate_headers",
			            "failed to create a valid seed block", false,
			            status_name(status));
			return;
		}
	}

	for (int i = 0; i < 50; i++)
	{
		size_t blocks = 1 + static_cast<size_t>(i % 12);
		ContextGuard context;
		astcenc_error status = make_context(
		    ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FASTEST,
		    ASTCENC_FLG_DECOMPRESS_ONLY | ASTCENC_FLG_SELF_DECOMPRESS_ONLY,
		    context);
		if (status != ASTCENC_SUCCESS)
		{
			record_case(151 + i, "duplicate_headers",
			            "decompress context setup failed", false,
			            status_name(status));
			continue;
		}

		std::vector<uint8_t> header_a;
		std::vector<uint8_t> header_b;
		append_header(header_a, 4, 4, 1, 4, 4, 1, ASTC_MAGIC);
		append_header(header_b,
		              (i % 2) ? 12 : 0,
		              (i % 3) ? 12 : 0,
		              (i % 5) ? 1 : 0,
		              0xFFFFFF, 0xFFFFFF, 0xFFFFFF,
		              (i % 4 == 0) ? 0xDEADBEEFu : ASTC_MAGIC);

		std::vector<uint8_t> payload;
		payload.reserve(blocks * 16);
		for (size_t b = 0; b < blocks; b++)
		{
			const uint8_t* chunk;
			if (b % 3 == 0 && i % 2 == 0)
			{
				chunk = header_a.data();
			}
			else if (b % 3 == 1 && i % 3 == 0)
			{
				chunk = header_b.data();
			}
			else if (b % 2 == 0)
			{
				chunk = valid_block;
			}
			else
			{
				chunk = header_b.data();
			}
			payload.insert(payload.end(), chunk, chunk + 16);
		}

		size_t dim_x = blocks * 4;
		size_t dim_y = 4;
		GuardedBuffer output(dim_x * dim_y * 4);
		astcenc_image image {};
		image.dim_x = dim_x;
		image.dim_y = dim_y;
		image.dim_z = 1;
		image.data_type = ASTCENC_TYPE_U8;
		image.data = nullptr;
		image.row_stride = 0;

		void* slice = output.data();
		image.data = &slice;

		status = astcenc_decompress_image(
		    context.context, payload.data(), payload.size(), &image,
		    &SWIZZLE_RGBA, 0);
		g_totals.api_calls++;

		bool block_info_ok = true;
		for (size_t b = 0; b < blocks; b++)
		{
			astcenc_block_info info {};
			astcenc_error info_status = astcenc_get_block_info(
			    context.context, payload.data() + b * 16, &info);
			g_totals.api_calls++;
			if (info_status != ASTCENC_SUCCESS)
			{
				block_info_ok = false;
				break;
			}
		}

		bool pass = is_block_decode_status(status)
		         && block_info_ok
		         && output.intact();

		char detail[512];
		std::snprintf(detail, sizeof(detail),
		              "blocks=%zu status=%s block_info=%s output_guard=%s",
		              blocks, status_name(status),
		              block_info_ok ? "ok" : "error",
		              output.intact() ? "ok" : "corrupt");

		record_case(151 + i, "duplicate_headers",
		            "alternating_headers_" + std::to_string(i),
		            pass, detail);
	}
}

//
// Legal inputs and reference encoding.
//

struct LegalSpec
{
	int id;
	size_t dim_x;
	size_t dim_y;
	size_t dim_z;
	unsigned int block_x;
	unsigned int block_y;
	unsigned int block_z;
	astcenc_profile profile;
	float quality;
	unsigned int flags;
	astcenc_swizzle swizzle;
	uint64_t seed;
};

std::vector<LegalSpec> make_legal_specs()
{
	static const std::array<std::array<unsigned int, 2>, 14> block_2d {{
		{{ 4, 4 }}, {{ 5, 4 }}, {{ 5, 5 }}, {{ 6, 5 }}, {{ 6, 6 }},
		{{ 8, 5 }}, {{ 8, 6 }}, {{ 8, 8 }}, {{ 10, 5 }}, {{ 10, 6 }},
		{{ 10, 8 }}, {{ 10, 10 }}, {{ 12, 10 }}, {{ 12, 12 }}
	}};

	static const std::array<std::array<unsigned int, 3>, 6> block_3d {{
		{{ 3, 3, 3 }}, {{ 4, 3, 3 }}, {{ 4, 4, 3 }},
		{{ 4, 4, 4 }}, {{ 5, 4, 4 }}, {{ 6, 6, 6 }}
	}};

	static const size_t dims_2d[] {
		1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 23, 31, 32, 33, 47
	};

	static const float qualities[] {
		ASTCENC_PRE_FASTEST, ASTCENC_PRE_FAST, ASTCENC_PRE_MEDIUM
	};

	std::vector<LegalSpec> specs;
	specs.reserve(50);

	for (int i = 0; i < 34; i++)
	{
		const auto& block = block_2d[i % block_2d.size()];
		size_t dx = dims_2d[(i * 5 + 3) % (sizeof(dims_2d) / sizeof(dims_2d[0]))];
		size_t dy = dims_2d[(i * 7 + 11) % (sizeof(dims_2d) / sizeof(dims_2d[0]))];

		// Keep high-effort cases small.
		if (i % 3 == 2)
		{
			dx = std::min<size_t>(dx, 17);
			dy = std::min<size_t>(dy, 13);
		}

		unsigned int flags = 0;
		astcenc_swizzle swizzle = SWIZZLE_RGBA;
		if (i % 5 == 1)
		{
			flags = ASTCENC_FLG_USE_DECODE_UNORM8;
		}
		else if (i % 5 == 2)
		{
			flags = ASTCENC_FLG_USE_ALPHA_WEIGHT;
		}
		else if (i % 5 == 3)
		{
			flags = ASTCENC_FLG_MAP_NORMAL;
			swizzle = SWIZZLE_RRRG;
		}
		else if (i % 5 == 4)
		{
			flags = ASTCENC_FLG_SELF_DECOMPRESS_ONLY;
		}

		specs.push_back({
		    i + 1, dx, dy, 1, block[0], block[1], 1,
		    (i % 2) ? ASTCENC_PRF_LDR : ASTCENC_PRF_LDR_SRGB,
		    qualities[i % (sizeof(qualities) / sizeof(qualities[0]))],
		    flags, swizzle, static_cast<uint64_t>(0x1000 + i)
		});
	}

	for (int i = 34; i < 50; i++)
	{
		const auto& block = block_3d[(i - 34) % block_3d.size()];
		size_t dx = 1 + static_cast<size_t>((i * 3 + 1) % 9);
		size_t dy = 1 + static_cast<size_t>((i * 5 + 2) % 7);
		size_t dz = 1 + static_cast<size_t>((i * 7 + 3) % 6);

		specs.push_back({
		    i + 1, dx, dy, dz, block[0], block[1], block[2],
		    ASTCENC_PRF_LDR, qualities[i % 3],
		    (i % 4 == 0) ? ASTCENC_FLG_USE_DECODE_UNORM8 : 0,
		    SWIZZLE_RGBA, static_cast<uint64_t>(0x2000 + i)
		});
	}

	return specs;
}

void fill_legal_image(ImageFixture& fixture, uint64_t seed)
{
	uint8_t* p = fixture.pixels.data();
	size_t texels = fixture.pixels.size() / 4;

	for (size_t i = 0; i < texels; i++)
	{
		uint8_t x = static_cast<uint8_t>(i);
		uint8_t y = static_cast<uint8_t>(i >> 8);
		uint8_t z = static_cast<uint8_t>(i >> 16);
		p[i * 4 + 0] = static_cast<uint8_t>(x * 13 + y * 3 + seed);
		p[i * 4 + 1] = static_cast<uint8_t>(x * 5 + y * 17 + z * 7 + seed);
		p[i * 4 + 2] = static_cast<uint8_t>(x * 11 + y * 19 + z * 23 + (seed >> 8));
		p[i * 4 + 3] = static_cast<uint8_t>((x ^ y ^ z) * 7 + 0x40);
	}
}

struct LegalEncoding
{
	std::vector<uint8_t> payload;
	std::string error;
};

LegalEncoding encode_legal(const LegalSpec& spec)
{
	LegalEncoding result;
	astcenc_config config {};
	astcenc_error status = astcenc_config_init(
	    spec.profile, spec.block_x, spec.block_y, spec.block_z,
	    spec.quality, spec.flags, &config);
	if (status != ASTCENC_SUCCESS)
	{
		result.error = std::string("config_init: ") + status_name(status);
		return result;
	}

	ContextGuard context;
	status = astcenc_context_alloc(&config, 1, &context.context, nullptr);
	if (status != ASTCENC_SUCCESS)
	{
		result.error = std::string("context_alloc: ") + status_name(status);
		return result;
	}

	ImageFixture fixture(spec.dim_x, spec.dim_y, spec.dim_z,
	                     ASTCENC_TYPE_U8, spec.seed);
	fill_legal_image(fixture, spec.seed);

	size_t blocks = block_count(
	    spec.dim_x, spec.dim_y, spec.dim_z,
	    spec.block_x, spec.block_y, spec.block_z);
	GuardedBuffer output(blocks * 16);

	status = astcenc_compress_image(
	    context.context, &fixture.image, &spec.swizzle,
	    output.data(), output.size(), 0);
	g_totals.api_calls++;

	if (status != ASTCENC_SUCCESS)
	{
		result.error = std::string("compress: ") + status_name(status);
		return result;
	}

	if (!fixture.pixels.intact() || !output.intact())
	{
		result.error = "guard corruption";
		return result;
	}

	result.payload.assign(output.data(), output.data() + output.size());
	return result;
}

void emit_header(std::vector<uint8_t>& out, const LegalSpec& spec)
{
	auto push_u24 = [&out](size_t value) {
		out.push_back(static_cast<uint8_t>(value));
		out.push_back(static_cast<uint8_t>(value >> 8));
		out.push_back(static_cast<uint8_t>(value >> 16));
	};

	out.push_back(static_cast<uint8_t>(ASTC_MAGIC));
	out.push_back(static_cast<uint8_t>(ASTC_MAGIC >> 8));
	out.push_back(static_cast<uint8_t>(ASTC_MAGIC >> 16));
	out.push_back(static_cast<uint8_t>(ASTC_MAGIC >> 24));
	out.push_back(static_cast<uint8_t>(spec.block_x));
	out.push_back(static_cast<uint8_t>(spec.block_y));
	out.push_back(static_cast<uint8_t>(spec.block_z));
	push_u24(spec.dim_x);
	push_u24(spec.dim_y);
	push_u24(spec.dim_z);
}

uint32_t read_u32(const uint8_t* p)
{
	return static_cast<uint32_t>(p[0])
	     | (static_cast<uint32_t>(p[1]) << 8)
	     | (static_cast<uint32_t>(p[2]) << 16)
	     | (static_cast<uint32_t>(p[3]) << 24);
}

size_t read_u24(const uint8_t* p)
{
	return static_cast<size_t>(p[0])
	     | (static_cast<size_t>(p[1]) << 8)
	     | (static_cast<size_t>(p[2]) << 16);
}

int emit_reference(const std::filesystem::path& dir)
{
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	if (ec)
	{
		std::fprintf(stderr, "ERROR: cannot create %s: %s\n",
		             dir.string().c_str(), ec.message().c_str());
		return 1;
	}

	std::ofstream manifest(dir / "manifest.tsv");
	manifest << "id\tdim_x\tdim_y\tdim_z\tblock_x\tblock_y\tblock_z\tprofile\tquality\tflags\tblocks\n";

	auto specs = make_legal_specs();
	int failures = 0;
	for (const LegalSpec& spec : specs)
	{
		LegalEncoding encoding = encode_legal(spec);
		if (!encoding.error.empty())
		{
			std::fprintf(stderr, "REFERENCE FAIL %d: %s\n",
			             spec.id, encoding.error.c_str());
			failures++;
			continue;
		}

		std::vector<uint8_t> file;
		file.reserve(16 + encoding.payload.size());
		emit_header(file, spec);
		file.insert(file.end(), encoding.payload.begin(), encoding.payload.end());

		char name[64];
		std::snprintf(name, sizeof(name), "case_%02d.astc", spec.id);
		std::ofstream out(dir / name, std::ios::binary);
		out.write(reinterpret_cast<const char*>(file.data()),
		          static_cast<std::streamsize>(file.size()));
		out.close();
		if (!out)
		{
			std::fprintf(stderr, "REFERENCE WRITE FAIL %d\n", spec.id);
			failures++;
			continue;
		}

		manifest << spec.id << '\t' << spec.dim_x << '\t' << spec.dim_y << '\t'
		         << spec.dim_z << '\t' << spec.block_x << '\t' << spec.block_y << '\t'
		         << spec.block_z << '\t' << static_cast<int>(spec.profile) << '\t'
		         << spec.quality << '\t' << spec.flags << '\t'
		         << encoding.payload.size() / 16 << '\n';
	}

	std::printf("REFERENCE cases=%zu failures=%d dir=%s\n",
	            specs.size(), failures, dir.string().c_str());
	return failures ? 1 : 0;
}

int verify_reference(const std::filesystem::path& dir)
{
	auto specs = make_legal_specs();
	for (const LegalSpec& spec : specs)
	{
		char name[64];
		std::snprintf(name, sizeof(name), "case_%02d.astc", spec.id);
		std::filesystem::path path = dir / name;

		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			record_case(spec.id, "legal_reference",
			            "missing reference file", false, path.string());
			continue;
		}

		std::vector<uint8_t> file(
		    (std::istreambuf_iterator<char>(in)),
		    std::istreambuf_iterator<char>());

		size_t expected_blocks = block_count(
		    spec.dim_x, spec.dim_y, spec.dim_z,
		    spec.block_x, spec.block_y, spec.block_z);
		size_t expected_size = 16 + expected_blocks * 16;

		bool header_ok = file.size() == expected_size
		              && file.size() >= 16
		              && read_u32(file.data()) == ASTC_MAGIC
		              && file[4] == spec.block_x
		              && file[5] == spec.block_y
		              && file[6] == spec.block_z
		              && read_u24(file.data() + 7) == spec.dim_x
		              && read_u24(file.data() + 10) == spec.dim_y
		              && read_u24(file.data() + 13) == spec.dim_z;

		if (!header_ok)
		{
			char detail[512];
			std::snprintf(detail, sizeof(detail),
			              "reference header/size mismatch actual=%zu expected=%zu",
			              file.size(), expected_size);
			record_case(spec.id, "legal_reference",
			            "reference file header", false, detail);
			continue;
		}

		LegalEncoding encoding = encode_legal(spec);
		g_totals.legal_cases++;
		g_totals.legal_blocks += expected_blocks;

		if (!encoding.error.empty())
		{
			record_case(spec.id, "legal_reference",
			            "re-encode failed", false, encoding.error);
			continue;
		}

		size_t mismatch = expected_blocks;
		for (size_t b = 0; b < expected_blocks; b++)
		{
			const uint8_t* expected = file.data() + 16 + b * 16;
			const uint8_t* actual = encoding.payload.data() + b * 16;
			if (std::memcmp(expected, actual, 16) != 0)
			{
				mismatch = b;
				break;
			}
		}

		if (mismatch != expected_blocks)
		{
			const uint8_t* expected = file.data() + 16 + mismatch * 16;
			const uint8_t* actual = encoding.payload.data() + mismatch * 16;
			char detail[1024];
			std::snprintf(detail, sizeof(detail),
			              "block=%zu/%zu expected=%s actual=%s dims=%zux%zux%zu block=%ux%ux%u q=%.1f flags=0x%x",
			              mismatch, expected_blocks,
			              hex_bytes(expected, 16).c_str(),
			              hex_bytes(actual, 16).c_str(),
			              spec.dim_x, spec.dim_y, spec.dim_z,
			              spec.block_x, spec.block_y, spec.block_z,
			              spec.quality, spec.flags);
			record_case(spec.id, "legal_reference",
			            "block mismatch", false, detail);
			g_totals.block_mismatches++;
		}
		else
		{
			record_case(spec.id, "legal_reference",
			            "all_blocks_match", true,
			            "blocks=" + std::to_string(expected_blocks));
		}
	}

	return g_totals.block_mismatches ? 1 : 0;
}

void print_results()
{
	for (const CaseResult& result : g_results)
	{
		std::printf("CASE\t%d\t%s\t%s\t%s\t%s\n",
		            result.id, result.category.c_str(), result.name.c_str(),
		            result.pass ? "PASS" : "FAIL", result.detail.c_str());
	}

	std::printf("SUMMARY\tcases=%zu\tpassed=%zu\tfailed=%zu\tapi_calls=%zu\t"
	            "legal_cases=%zu\tlegal_blocks=%zu\tblock_mismatches=%zu\n",
	            g_totals.cases, g_totals.passed, g_totals.failed,
	            g_totals.api_calls, g_totals.legal_cases,
	            g_totals.legal_blocks, g_totals.block_mismatches);
}

void usage(const char* argv0)
{
	std::fprintf(stderr,
	             "Usage: %s --illegal | --emit-reference DIR | --verify-reference DIR\n",
	             argv0);
}

} // namespace

int main(int argc, char** argv)
{
	if (argc != 2 && argc != 3)
	{
		usage(argv[0]);
		return 1;
	}

	std::string mode = argv[1];
	int rc = 0;

	if (mode == "--illegal")
	{
		run_negative_dimension_cases();
		run_block_geometry_cases();
		run_oversized_quality_cases();
		run_duplicate_header_cases();
	}
	else if (mode == "--emit-reference" && argc == 3)
	{
		rc = emit_reference(argv[2]);
		return rc;
	}
	else if (mode == "--verify-reference" && argc == 3)
	{
		rc = verify_reference(argv[2]);
	}
	else
	{
		usage(argv[0]);
		return 1;
	}

	print_results();
	return (g_totals.failed || rc) ? 1 : 0;
}
