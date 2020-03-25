// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "ocv_common.hpp"

#include <inference_engine.hpp>

/**
 * @brief Base class of config for network
 */
struct CnnConfig {
	explicit CnnConfig(const std::string& path_to_model,
		const std::string& path_to_weights)
		: path_to_model(path_to_model), path_to_weights(path_to_weights) {}

	/** @brief Path to model description */
	std::string path_to_model;
	/** @brief Path to model weights */
	std::string path_to_weights;
	/** @brief Maximal size of batch */
	int max_batch_size{ 1 };
};
