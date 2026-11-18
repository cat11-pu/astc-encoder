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
 * @brief Unit tests for strided image storage.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "../astcenc.h"

namespace astcenc
{

/** @brief An owning pointer to a codec context. */
using astcenc_context_unq_ptr = std::unique_ptr<astcenc_context, decltype(&astcenc_context_free)>;

/**
 * @brief Get the number of blocks needed to cover an axis.
 *
 * @param dim     The axis dimension, in texels.
 * @param block   The block dimension, in texels.
 *
 * @return The number of blocks.
 */
static size_t get_block_count(size_t dim, size_t block)
{
	return (dim + block - 1) / block;
}

/**
 * @brief Get the compressed byte count for a region.
 *
 * @param region   The region.
 * @param block_x  The block X dimension.
 * @param block_y  The block Y dimension.
 * @param block_z  The block Z dimension.
 *
 * @return The number of compressed bytes.
 */
static size_t get_region_byte_count(
	const astcenc_region& region,
	size_t block_x,
	size_t block_y,
	size_t block_z
) {
	return get_block_count(region.dim_x, block_x)
	     * get_block_count(region.dim_y, block_y)
	     * get_block_count(region.dim_z, block_z) * 16;
}

/**
 * @brief Compare a region encode against the corresponding blocks from a complete-image encode.
 *
 * @param context            The codec context.
 * @param image              The complete image.
 * @param region             The region to encode.
 * @param swizzle            The compression swizzle.
 * @param block_x            The block X dimension.
 * @param block_y            The block Y dimension.
 * @param block_z            The block Z dimension.
 * @param full_compressed    The complete-image compressed data.
 * @param[out] region_bytes  The region compressed data.
 *
 * @return @c true if every block is byte-identical, @c false otherwise.
 */
static bool compare_region_blocks(
	astcenc_context* context,
	astcenc_image& image,
	const astcenc_region& region,
	const astcenc_swizzle& swizzle,
	size_t block_x,
	size_t block_y,
	size_t block_z,
	const std::vector<uint8_t>& full_compressed,
	std::vector<uint8_t>& region_bytes
) {
	size_t byte_count = get_region_byte_count(region, block_x, block_y, block_z);
	region_bytes.resize(byte_count);

	astcenc_error status = astcenc_compress_image_region(
		context, &image, &region, &swizzle, region_bytes.data(), region_bytes.size(), 0);
	if (status != ASTCENC_SUCCESS)
	{
		ADD_FAILURE() << "Region compression failed with status " << status;
		return false;
	}

	size_t full_blocks_x = get_block_count(image.dim_x, block_x);
	size_t full_blocks_y = get_block_count(image.dim_y, block_y);
	size_t region_blocks_x = get_block_count(region.dim_x, block_x);
	size_t region_blocks_y = get_block_count(region.dim_y, block_y);
	size_t region_blocks_z = get_block_count(region.dim_z, block_z);

	size_t origin_block_x = region.origin_x / block_x;
	size_t origin_block_y = region.origin_y / block_y;
	size_t origin_block_z = region.origin_z / block_z;

	for (size_t z = 0; z < region_blocks_z; z++)
	{
		for (size_t y = 0; y < region_blocks_y; y++)
		{
			for (size_t x = 0; x < region_blocks_x; x++)
			{
				size_t region_index = ((z * region_blocks_y + y) * region_blocks_x + x) * 16;
				size_t full_index = ((((z + origin_block_z) * full_blocks_y) +
				                      (y + origin_block_y)) * full_blocks_x +
				                     (x + origin_block_x)) * 16;

				if (std::memcmp(region_bytes.data() + region_index,
				                full_compressed.data() + full_index, 16) != 0)
				{
					ADD_FAILURE() << "Block mismatch at region block (" << x << ", "
					              << y << ", " << z << ")";
					return false;
				}
			}
		}
	}

	return true;
}

/**
 * @brief Compare packed and strided compression and decompression.
 *
 * @tparam T                 The component storage type.
 *
 * @param  type              The component format.
 * @param  depth             The number of image planes.
 * @param  is_alpha_scaled   Whether to exercise swizzling and alpha preprocessing.
 */
template<typename T>
static void test_row_stride(
	astcenc_type type,
	size_t depth,
	bool is_alpha_scaled
) {
	constexpr size_t width = 5;
	constexpr size_t height = 7;
	constexpr size_t row_components = 4 * width + 3;
	constexpr size_t offset = row_components + 4;
	constexpr size_t slice_components = offset + height * row_components + 4;
	constexpr T sentinel = static_cast<T>(42);

	std::vector<T> packed(width * height * depth * 4);
	std::vector<T> strided(slice_components * depth, sentinel);
	std::vector<void*> packed_slices(depth);
	std::vector<void*> strided_slices(depth);
	for (size_t z = 0; z < depth; z++)
	{
		packed_slices[z] = packed.data() + z * width * height * 4;
		strided_slices[z] = strided.data() + z * slice_components + offset;
		for (size_t y = 0; y < height; y++)
		{
			for (size_t x = 0; x < 4 * width; x++)
			{
				size_t value = (x * 3 + y * 2 + z * 7) % 5;
				T component;
				if constexpr (std::is_same_v<T, uint16_t>)
				{
					constexpr uint16_t values[] { 0, 0x3400, 0x3800, 0x3A00, 0x3C00 };
					component = values[value];
				}
				else if constexpr (std::is_same_v<T, uint8_t>)
				{
					component = static_cast<T>(value * 63);
				}
				else
				{
					component = static_cast<T>(value) * 0.25f;
				}

				packed[(z * height + y) * width * 4 + x] = component;
				strided[z * slice_components + offset + y * row_components + x] = component;
			}
		}
	}

	astcenc_image packed_image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = depth,
		.data_type = type,
		.data = packed_slices.data(),
		.row_stride = 0
	};

	astcenc_image strided_image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = depth,
		.data_type = type,
		.data = strided_slices.data(),
		.row_stride = row_components * sizeof(T)
	};

	astcenc_config config;
	unsigned int block_xy = 3;
	unsigned int block_z = 3;
	if (depth == 1)
	{
		block_xy = 4;
		block_z = 1;
	}

	astcenc_error status = astcenc_config_init(
	    ASTCENC_PRF_LDR, block_xy, block_xy, block_z, ASTCENC_PRE_FASTEST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	if (is_alpha_scaled)
	{
		config.a_scale_radius = 1;
	}

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	astcenc_swizzle swizzle {
		.r = ASTCENC_SWZ_R,
		.g = ASTCENC_SWZ_G,
		.b = ASTCENC_SWZ_B,
		.a = ASTCENC_SWZ_A
	};

	if (is_alpha_scaled)
	{
		swizzle.r = ASTCENC_SWZ_B;
		swizzle.b = ASTCENC_SWZ_R;
	}

	size_t compressed_size = ((width + block_xy - 1) / block_xy)
	                       * ((height + block_xy - 1) / block_xy) * 16;
	std::vector<uint8_t> packed_compressed(compressed_size);
	std::vector<uint8_t> strided_compressed(compressed_size);

	status = astcenc_compress_image(context.get(), &packed_image, &swizzle,
	                                packed_compressed.data(), compressed_size, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(packed_image.row_stride, 0u);

	status = astcenc_compress_reset(context.get());
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	status = astcenc_compress_image(context.get(), &strided_image, &swizzle,
	                                strided_compressed.data(), compressed_size, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(strided_compressed, packed_compressed);

	// Explicit packed pitch must match the zero-stride default
	packed_image.row_stride = width * 4 * sizeof(T);
	status = astcenc_compress_reset(context.get());
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	status = astcenc_compress_image(context.get(), &packed_image, &swizzle,
	                                packed_compressed.data(), compressed_size, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(strided_compressed, packed_compressed);

	std::fill(strided.begin(), strided.end(), sentinel);
	packed_image.row_stride = 0;

	status = astcenc_decompress_image(context.get(), packed_compressed.data(), compressed_size,
	                                  &packed_image, &swizzle, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(packed_image.row_stride, 0u);

	status = astcenc_decompress_reset(context.get());
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	status = astcenc_decompress_image(context.get(), packed_compressed.data(), compressed_size,
	                                  &strided_image, &swizzle, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	std::vector<T> expected(strided.size(), sentinel);
	for (size_t z = 0; z < depth; z++)
	{
		for (size_t y = 0; y < height; y++)
		{
			std::copy_n(packed.data() + (z * height + y) * width * 4, width * 4,
			            expected.data() + z * slice_components + offset + y * row_components);
		}
	}
	EXPECT_EQ(strided, expected);
}

/** @brief Row stride supports every component type, cropped origins, and 3D images. */
TEST(image, row_stride)
{
	for (size_t depth : { 1u, 3u })
	{
		for (bool is_alpha_scaled : { false, true })
		{
			SCOPED_TRACE(::testing::Message() << "Depth " << depth << ", alpha " << is_alpha_scaled);
			test_row_stride<uint8_t>(ASTCENC_TYPE_U8, depth, is_alpha_scaled);
			test_row_stride<uint16_t>(ASTCENC_TYPE_F16, depth, is_alpha_scaled);
			test_row_stride<float>(ASTCENC_TYPE_F32, depth, is_alpha_scaled);
		}
	}
}

/** @brief Invalid strides must be rejected before accessing image storage. */
TEST(image, invalid_row_stride)
{
	astcenc_config config;
	astcenc_error status = astcenc_config_init(
	    ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FASTEST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	for (size_t component_size : { 1u, 2u, 4u })
	{
		astcenc_type type = ASTCENC_TYPE_F32;
		if (component_size == 1)
		{
			type = ASTCENC_TYPE_U8;
		}
		else if (component_size == 2)
		{
			type = ASTCENC_TYPE_F16;
		}

		size_t row_bytes = 4 * 4 * component_size;
		size_t max_size = std::numeric_limits<size_t>::max();
		const size_t strides[] {
			row_bytes - 1,
			max_size - component_size + 1,
			(max_size / 2 / component_size) * component_size
		};

		for (size_t stride : strides)
		{
			astcenc_image image {
				.dim_x = 4,
				.dim_y = 4,
				.dim_z = 1,
				.data_type = type,
				.data = nullptr,
				.row_stride = stride
			};

			status = astcenc_compress_image(context.get(), &image, &swizzle, nullptr, 0, 0);
			EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

			status = astcenc_decompress_image(context.get(), nullptr, 0, &image, &swizzle, 0);
			EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);
		}

		// Two rows can overflow only when adding the final row width
		astcenc_image bad_image {
			.dim_x = 4,
			.dim_y = 2,
			.dim_z = 1,
			.data_type = type,
			.data = nullptr,
			.row_stride = (max_size / component_size) * component_size
		};

		status = astcenc_compress_image(context.get(), &bad_image, &swizzle, nullptr, 0, 0);
		EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

		status = astcenc_decompress_image(context.get(), nullptr, 0, &bad_image, &swizzle, 0);
		EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

		if (component_size > 1)
		{
			astcenc_image image {
				.dim_x = 4,
				.dim_y = 1,
				.dim_z = 1,
				.data_type = type,
				.data = nullptr,
				.row_stride = row_bytes + 1
			};

			status = astcenc_compress_image(context.get(), &image, &swizzle, nullptr, 0, 0);
			EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

			status = astcenc_decompress_image(context.get(), nullptr, 0, &image, &swizzle, 0);
			EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);
		}
	}
}

/** @brief Region compression must match complete-image compression block for block in 2D. */
TEST(region, matches_full_image_crop_2d)
{
	constexpr size_t width = 37;
	constexpr size_t height = 29;
	constexpr unsigned int block_size = 6;
	constexpr size_t row_stride = width * 4 + 7;
	constexpr size_t pixel_offset = 3;

	std::vector<uint8_t> pixels(pixel_offset + height * row_stride, 0xA5);
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			size_t index = pixel_offset + y * row_stride + x * 4;
			pixels[index + 0] = static_cast<uint8_t>((x * 17 + y * 3) & 0xFF);
			pixels[index + 1] = static_cast<uint8_t>((x * 5 + y * 29) & 0xFF);
			pixels[index + 2] = static_cast<uint8_t>((x * 11 + y * 7) & 0xFF);
			pixels[index + 3] = (x + y) % 13 == 0 ? 0 : static_cast<uint8_t>((x * 13 + y) & 0xFF);
		}
	}

	void* slices = pixels.data() + pixel_offset;
	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &slices,
		.row_stride = row_stride
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
		ASTCENC_PRF_LDR, block_size, block_size, 1, ASTCENC_PRE_FASTEST,
		ASTCENC_FLG_USE_ALPHA_WEIGHT, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	config.a_scale_radius = 1;

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	size_t full_byte_count = get_block_count(width, block_size)
	                       * get_block_count(height, block_size) * 16;
	std::vector<uint8_t> full_compressed(full_byte_count);
	status = astcenc_compress_image(
		context.get(), &image, &swizzle, full_compressed.data(), full_compressed.size(), 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	const astcenc_region regions[] {
		{ 0, 0, 0, 12, 12, 1 },
		{ 18, 6, 0, 18, 18, 1 },
		{ 6, 0, 0, 6, 6, 1 },
		{ 0, 24, 0, 6, 5, 1 },
		{ 36, 24, 0, 1, 5, 1 },
		{ 30, 0, 0, 7, 12, 1 },
		{ 0, 0, 0, width, height, 1 }
	};

	for (const astcenc_region& region : regions)
	{
		SCOPED_TRACE(::testing::Message() << "Origin " << region.origin_x << ","
		              << region.origin_y << " size " << region.dim_x << "x" << region.dim_y);
		std::vector<uint8_t> region_compressed;
		EXPECT_TRUE(compare_region_blocks(
			context.get(), image, region, swizzle, block_size, block_size, 1,
			full_compressed, region_compressed));
	}

	// The standalone region stream must decode as a texture with the region dimensions.
	astcenc_region decode_region { 36, 24, 0, 1, 5, 1 };
	std::vector<uint8_t> region_compressed;
	ASSERT_TRUE(compare_region_blocks(
		context.get(), image, decode_region, swizzle, block_size, block_size, 1,
		full_compressed, region_compressed));

	std::vector<uint8_t> full_decoded(width * height * 4);
	void* full_decoded_slice = full_decoded.data();
	astcenc_image full_decoded_image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &full_decoded_slice,
		.row_stride = 0
	};

	status = astcenc_decompress_image(
		context.get(), full_compressed.data(), full_compressed.size(),
		&full_decoded_image, &swizzle, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	std::vector<uint8_t> region_decoded(decode_region.dim_x * decode_region.dim_y * 4);
	void* region_decoded_slice = region_decoded.data();
	astcenc_image region_decoded_image {
		.dim_x = decode_region.dim_x,
		.dim_y = decode_region.dim_y,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &region_decoded_slice,
		.row_stride = 0
	};

	status = astcenc_decompress_image(
		context.get(), region_compressed.data(), region_compressed.size(),
		&region_decoded_image, &swizzle, 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	for (size_t y = 0; y < decode_region.dim_y; y++)
	{
		size_t full_offset = ((y + decode_region.origin_y) * width +
		                      decode_region.origin_x) * 4;
		size_t region_offset = y * decode_region.dim_x * 4;
		EXPECT_EQ(std::memcmp(region_decoded.data() + region_offset,
		                      full_decoded.data() + full_offset,
		                      decode_region.dim_x * 4), 0);
	}
}

/** @brief Region compression must match complete-image compression block for block in 3D. */
TEST(region, matches_full_image_crop_3d)
{
	constexpr size_t width = 11;
	constexpr size_t height = 10;
	constexpr size_t depth = 8;
	constexpr unsigned int block_size = 3;

	std::vector<uint8_t> pixels(width * height * depth * 4);
	for (size_t z = 0; z < depth; z++)
	{
		for (size_t y = 0; y < height; y++)
		{
			for (size_t x = 0; x < width; x++)
			{
				size_t index = ((z * height + y) * width + x) * 4;
				pixels[index + 0] = static_cast<uint8_t>((x * 17 + y * 3 + z * 7) & 0xFF);
				pixels[index + 1] = static_cast<uint8_t>((x * 5 + y * 29 + z * 11) & 0xFF);
				pixels[index + 2] = static_cast<uint8_t>((x * 11 + y * 7 + z * 13) & 0xFF);
				pixels[index + 3] = static_cast<uint8_t>((x * 13 + y + z * 19) & 0xFF);
			}
		}
	}

	std::vector<void*> slices(depth);
	size_t plane_size = width * height * 4;
	for (size_t z = 0; z < depth; z++)
	{
		slices[z] = pixels.data() + z * plane_size;
	}

	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = depth,
		.data_type = ASTCENC_TYPE_U8,
		.data = slices.data(),
		.row_stride = 0
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
		ASTCENC_PRF_LDR, block_size, block_size, block_size,
		ASTCENC_PRE_FASTEST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	size_t full_byte_count = get_block_count(width, block_size)
	                       * get_block_count(height, block_size)
	                       * get_block_count(depth, block_size) * 16;
	std::vector<uint8_t> full_compressed(full_byte_count);
	status = astcenc_compress_image(
		context.get(), &image, &swizzle, full_compressed.data(), full_compressed.size(), 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	const astcenc_region regions[] {
		{ 3, 3, 3, 3, 3, 3 },
		{ 0, 0, 0, 6, 6, 3 },
		{ 9, 9, 6, 2, 1, 2 },
		{ 0, 0, 0, width, height, depth }
	};

	for (const astcenc_region& region : regions)
	{
		SCOPED_TRACE(::testing::Message() << "Origin " << region.origin_x << ","
		              << region.origin_y << "," << region.origin_z << " size "
		              << region.dim_x << "x" << region.dim_y << "x" << region.dim_z);
		std::vector<uint8_t> region_compressed;
		EXPECT_TRUE(compare_region_blocks(
			context.get(), image, region, swizzle, block_size, block_size, block_size,
			full_compressed, region_compressed));
	}
}

/** @brief Invalid region extents and buffer sizes must fail before touching source data. */
TEST(region, invalid_regions)
{
	constexpr size_t width = 16;
	constexpr size_t height = 16;
	constexpr unsigned int block_size = 4;

	std::vector<uint8_t> pixels(width * height * 4);
	void* slices = pixels.data();
	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &slices,
		.row_stride = 0
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
		ASTCENC_PRF_LDR, block_size, block_size, 1,
		ASTCENC_PRE_FASTEST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	uint8_t data[16];
	const astcenc_region invalid_regions[] {
		{ 1, 0, 0, 4, 4, 1 },
		{ 0, 2, 0, 4, 4, 1 },
		{ 4, 4, 0, 3, 4, 1 },
		{ 4, 4, 0, 4, 3, 1 },
		{ 15, 0, 0, 1, 4, 1 },
		{ 16, 0, 0, 1, 4, 1 },
		{ 0, 0, 0, 0, 4, 1 },
		{ 0, 0, 0, 4, 0, 1 },
		{ 0, 0, 0, 20, 4, 1 },
		{ 0, 0, 0, std::numeric_limits<size_t>::max(), 4, 1 }
	};

	for (const astcenc_region& region : invalid_regions)
	{
		status = astcenc_compress_image_region(
			context.get(), &image, &region, &swizzle, data, sizeof(data), 0);
		EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);
	}

	astcenc_region valid_region { 0, 0, 0, 4, 4, 1 };
	status = astcenc_compress_image_region(
		context.get(), &image, &valid_region, &swizzle, data, sizeof(data) - 1, 0);
	EXPECT_EQ(status, ASTCENC_ERR_OUT_OF_MEM);

	status = astcenc_compress_image_region(
		context.get(), &image, &valid_region, &swizzle, nullptr, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

	astcenc_image image_without_data = image;
	image_without_data.data = nullptr;
	status = astcenc_compress_image_region(
		context.get(), &image_without_data, &valid_region, &swizzle, data, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

	status = astcenc_compress_image_region(
		nullptr, &image, &valid_region, &swizzle, data, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

	status = astcenc_compress_image_region(
		context.get(), nullptr, &valid_region, &swizzle, data, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

	status = astcenc_compress_image_region(
		context.get(), &image, nullptr, &swizzle, data, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);

	status = astcenc_compress_image_region(
		context.get(), &image, &valid_region, nullptr, data, sizeof(data), 0);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);
}

/** @brief A multi-threaded region encode must match a multi-threaded complete-image encode. */
TEST(region, threaded_matches_full_image_crop)
{
	constexpr size_t width = 20;
	constexpr size_t height = 20;
	constexpr size_t thread_count = 2;
	constexpr unsigned int block_size = 4;

	std::vector<uint8_t> pixels(width * height * 4);
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			size_t index = (y * width + x) * 4;
			pixels[index + 0] = static_cast<uint8_t>((x * 19 + y * 3) & 0xFF);
			pixels[index + 1] = static_cast<uint8_t>((x * 7 + y * 23) & 0xFF);
			pixels[index + 2] = static_cast<uint8_t>((x * 11 + y * 13) & 0xFF);
			pixels[index + 3] = static_cast<uint8_t>((x * 5 + y * 17) & 0xFF);
		}
	}

	void* slices = pixels.data();
	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &slices,
		.row_stride = 0
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
		ASTCENC_PRF_LDR, block_size, block_size, 1,
		ASTCENC_PRE_FASTEST, ASTCENC_FLG_USE_ALPHA_WEIGHT, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	config.a_scale_radius = 1;

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, thread_count, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	size_t full_byte_count = get_block_count(width, block_size)
	                       * get_block_count(height, block_size) * 16;
	std::vector<uint8_t> full_compressed(full_byte_count);
	std::vector<astcenc_error> statuses(thread_count, ASTCENC_SUCCESS);
	std::vector<std::thread> workers;
	for (size_t thread = 0; thread < thread_count; thread++)
	{
		workers.emplace_back([&, thread]() {
			statuses[thread] = astcenc_compress_image(
				context.get(), &image, &swizzle,
				full_compressed.data(), full_compressed.size(), thread);
		});
	}
	for (std::thread& worker : workers)
	{
		worker.join();
	}

	for (astcenc_error thread_status : statuses)
	{
		ASSERT_EQ(thread_status, ASTCENC_SUCCESS);
	}
	ASSERT_EQ(astcenc_compress_reset(context.get()), ASTCENC_SUCCESS);

	astcenc_region region { 4, 8, 0, 12, 8, 1 };
	size_t region_byte_count = get_region_byte_count(region, block_size, block_size, 1);
	std::vector<uint8_t> region_compressed(region_byte_count);
	std::fill(statuses.begin(), statuses.end(), ASTCENC_SUCCESS);

	workers.clear();
	for (size_t thread = 0; thread < thread_count; thread++)
	{
		workers.emplace_back([&, thread]() {
			statuses[thread] = astcenc_compress_image_region(
				context.get(), &image, &region, &swizzle,
				region_compressed.data(), region_compressed.size(), thread);
		});
	}
	for (std::thread& worker : workers)
	{
		worker.join();
	}

	for (astcenc_error thread_status : statuses)
	{
		ASSERT_EQ(thread_status, ASTCENC_SUCCESS);
	}

	std::vector<uint8_t> expected;
	size_t region_blocks_x = get_block_count(region.dim_x, block_size);
	size_t region_blocks_y = get_block_count(region.dim_y, block_size);
	size_t full_blocks_x = get_block_count(width, block_size);
	for (size_t y = 0; y < region_blocks_y; y++)
	{
		for (size_t x = 0; x < region_blocks_x; x++)
		{
			size_t full_index = (((region.origin_y / block_size + y) * full_blocks_x) +
			                     (region.origin_x / block_size + x)) * 16;
			expected.insert(expected.end(),
			                full_compressed.begin() + full_index,
			                full_compressed.begin() + full_index + 16);
		}
	}
	EXPECT_EQ(region_compressed, expected);

	ASSERT_EQ(astcenc_compress_reset(context.get()), ASTCENC_SUCCESS);
}

/** @brief Ten thousand region encodes must reuse one context without state drift. */
TEST(region, batch_10000_blocks)
{
	constexpr size_t width = 8;
	constexpr size_t height = 8;
	constexpr unsigned int block_size = 4;
	constexpr size_t blocks_x = width / block_size;
	constexpr size_t blocks_y = height / block_size;

	std::vector<uint8_t> pixels(width * height * 4);
	for (size_t i = 0; i < pixels.size(); i++)
	{
		pixels[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
	}

	void* slices = pixels.data();
	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = &slices,
		.row_stride = 0
	};

	astcenc_config config;
	astcenc_error status = astcenc_config_init(
		ASTCENC_PRF_LDR, block_size, block_size, 1,
		ASTCENC_PRE_FASTEST, ASTCENC_FLG_USE_ALPHA_WEIGHT, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	config.a_scale_radius = 1;

	astcenc_context* context_raw;
	status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

	const astcenc_swizzle swizzle {
		ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
	};

	std::vector<uint8_t> full_compressed(blocks_x * blocks_y * 16);
	status = astcenc_compress_image(
		context.get(), &image, &swizzle, full_compressed.data(), full_compressed.size(), 0);
	ASSERT_EQ(status, ASTCENC_SUCCESS);

	uint8_t compressed[16];
	for (size_t i = 0; i < 10000; i++)
	{
		size_t block = i % (blocks_x * blocks_y);
		astcenc_region region {
			(block % blocks_x) * block_size,
			(block / blocks_x) * block_size,
			0,
			block_size,
			block_size,
			1
		};

		status = astcenc_compress_image_region(
			context.get(), &image, &region, &swizzle, compressed, sizeof(compressed), 0);
		if (status != ASTCENC_SUCCESS ||
		    std::memcmp(compressed, full_compressed.data() + block * 16, 16) != 0)
		{
			ADD_FAILURE() << "Region encode failed or drifted at iteration " << i;
			break;
		}
	}
}

}
