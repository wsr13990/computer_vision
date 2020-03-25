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

/**
 * @brief Base class of network
 */
class CnnBase {
public:
	using Config = CnnConfig;

	/**
	 * @brief Constructor
	 */
	CnnBase(const Config& config,
		const InferenceEngine::Core& ie,
		const std::string& deviceName);

	/**
	 * @brief Descructor
	 */
	virtual ~CnnBase() {}

	/**
	 * @brief Loads network
	 */
	void Load();

	/**
	 * @brief Prints performance report
	 */
	void PrintPerformanceCounts(std::string fullDeviceName) const;
	/**
	 * @brief Run network in batch mode
	 *
	 * @param frames Vector of input images
	 * @param results_fetcher Callback to fetch inference results
	 */
	std::vector<float> Infer(const cv::Mat& frame) const;

protected:
	/** @brief Config */
	Config config_;
	/** @brief Inference Engine instance */
	InferenceEngine::Core ie_;
	/** @brief Inference Engine device */
	std::string deviceName_;
	/** @brief Net outputs info */
	InferenceEngine::OutputsDataMap outInfo_;
	/** @brief IE network */
	InferenceEngine::ExecutableNetwork executable_network_;
	/** @brief IE InferRequest */
	mutable InferenceEngine::InferRequest infer_request_;
	/** @brief Pointer to the pre-allocated input blob */
	mutable InferenceEngine::Blob::Ptr input_blob_;
	/** @brief Map of output blobs */
	InferenceEngine::BlobMap outputs_;
	size_t output_dimention;
};