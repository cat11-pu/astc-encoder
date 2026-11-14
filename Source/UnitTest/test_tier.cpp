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
 * @brief Unit tests for the quality tier interface.
 */

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "../astcenc.h"

namespace astcenc
{

/** @brief An owning pointer to a codec context. */
using astcenc_context_unq_ptr = std::unique_ptr<astcenc_context, decltype(&astcenc_context_free)>;

/** @brief Test that each tier selects its documented block size. */
TEST(tier, config_block_sizes)
{
	astcenc_config config;

	astcenc_error status = astcenc_tier_config_init(
	    ASTCENC_PRF_LDR, ASTCENC_TIER_FAST, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(config.block_x, 8u);
	EXPECT_EQ(config.block_y, 8u);
	EXPECT_EQ(config.block_z, 1u);

	status = astcenc_tier_config_init(
	    ASTCENC_PRF_LDR, ASTCENC_TIER_MEDIUM, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(config.block_x, 6u);
	EXPECT_EQ(config.block_y, 6u);
	EXPECT_EQ(config.block_z, 1u);

	status = astcenc_tier_config_init(
	    ASTCENC_PRF_LDR, ASTCENC_TIER_HIGH, 0, &config);
	ASSERT_EQ(status, ASTCENC_SUCCESS);
	EXPECT_EQ(config.block_x, 4u);
	EXPECT_EQ(config.block_y, 4u);
	EXPECT_EQ(config.block_z, 1u);
}

/** @brief Test that an invalid tier is rejected. */
TEST(tier, config_bad_tier)
{
	astcenc_config config;
	astcenc_error status = astcenc_tier_config_init(
	    ASTCENC_PRF_LDR, static_cast<astcenc_quality_tier>(42), 0, &config);
	EXPECT_EQ(status, ASTCENC_ERR_BAD_PARAM);
}

/**
 * @brief Test that every tier round-trips through the codec deterministically.
 *
 * Compressed output from every tier must decompress successfully, and two
 * decompressions of the same data must give bit-identical pixels.
 */
TEST(tier, roundtrip_deterministic)
{
	constexpr size_t width = 16;
	constexpr size_t height = 16;

	std::vector<uint8_t> image_data(width * height * 4);
	for (size_t i = 0; i < image_data.size(); i++)
	{
		image_data[i] = static_cast<uint8_t>((i * 7 + 13) % 251);
	}

	void* slices[1] { image_data.data() };
	astcenc_image image {
		.dim_x = width,
		.dim_y = height,
		.dim_z = 1,
		.data_type = ASTCENC_TYPE_U8,
		.data = slices,
		.row_stride = 0
	};

	astcenc_swizzle swizzle {
		.r = ASTCENC_SWZ_R,
		.g = ASTCENC_SWZ_G,
		.b = ASTCENC_SWZ_B,
		.a = ASTCENC_SWZ_A
	};

	const astcenc_quality_tier tiers[3] {
		ASTCENC_TIER_FAST, ASTCENC_TIER_MEDIUM, ASTCENC_TIER_HIGH
	};

	for (astcenc_quality_tier tier : tiers)
	{
		astcenc_config config;
		astcenc_error status = astcenc_tier_config_init(
		    ASTCENC_PRF_LDR, tier, 0, &config);
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		astcenc_context* context_raw;
		status = astcenc_context_alloc(&config, 1, &context_raw, nullptr);
		ASSERT_EQ(status, ASTCENC_SUCCESS);
		astcenc_context_unq_ptr context(context_raw, astcenc_context_free);

		size_t blocks_x = (width + config.block_x - 1) / config.block_x;
		size_t blocks_y = (height + config.block_y - 1) / config.block_y;
		size_t compressed_size = blocks_x * blocks_y * 16;
		std::vector<uint8_t> compressed(compressed_size);

		status = astcenc_compress_image(context.get(), &image, &swizzle,
		                                compressed.data(), compressed_size, 0);
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		std::vector<uint8_t> decode1(image_data.size());
		std::vector<uint8_t> decode2(image_data.size());
		void* decode1_slices[1] { decode1.data() };
		void* decode2_slices[1] { decode2.data() };
		astcenc_image decode1_image = image;
		astcenc_image decode2_image = image;
		decode1_image.data = decode1_slices;
		decode2_image.data = decode2_slices;

		status = astcenc_decompress_image(context.get(), compressed.data(),
		                                  compressed_size, &decode1_image,
		                                  &swizzle, 0);
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		status = astcenc_decompress_reset(context.get());
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		status = astcenc_decompress_image(context.get(), compressed.data(),
		                                  compressed_size, &decode2_image,
		                                  &swizzle, 0);
		ASSERT_EQ(status, ASTCENC_SUCCESS);

		EXPECT_EQ(decode1, decode2);
	}
}

}
