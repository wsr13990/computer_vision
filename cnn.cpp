// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;

CnnBase::CnnBase(const Config& config,
	const InferenceEngine::Core& ie,
	const std::string& deviceName) :
	config_(config), ie_(ie), deviceName_(deviceName) {}

void CnnBase::Load() {
	CNNNetReader netReader;
	netReader.ReadNetwork(config_.path_to_model);
	netReader.ReadWeights(config_.path_to_weights);
	auto cnnNetwork = netReader.getNetwork();

	const int currentBatchSize = cnnNetwork.getBatchSize();
	if (currentBatchSize != config_.max_batch_size)
		cnnNetwork.setBatchSize(config_.max_batch_size);

	InferenceEngine::InputsDataMap in;
	in = cnnNetwork.getInputsInfo();
	if (in.size() != 1) {
		THROW_IE_EXCEPTION << "Network should have only one input";
	}

	SizeVector inputDims = in.begin()->second->getTensorDesc().getDims();
	in.begin()->second->setPrecision(Precision::U8);
	input_blob_ = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, inputDims, Layout::NCHW));
	input_blob_->allocate();
	BlobMap inputs;
	inputs[in.begin()->first] = input_blob_;
	outInfo_ = cnnNetwork.getOutputsInfo();

	for (auto&& item : outInfo_) {
		SizeVector outputDims = item.second->getTensorDesc().getDims();
		auto outputLayout = item.second->getTensorDesc().getLayout();
		item.second->setPrecision(Precision::FP32);
		TBlob<float>::Ptr output =
			make_shared_blob<float>(TensorDesc(Precision::FP32, outputDims, outputLayout));
		output->allocate();
		outputs_[item.first] = output;
	}

	executable_network_ = ie_.LoadNetwork(cnnNetwork, deviceName_);
	infer_request_ = executable_network_.CreateInferRequest();
	infer_request_.SetInput(inputs);
	infer_request_.SetOutput(outputs_);
	const SizeVector outputDim = outInfo_.begin()->second->getTensorDesc().getDims();
	output_dimention = outputDim[1];
}

std::vector<float> CnnBase::Infer(
	const cv::Mat& frame) const {
	const size_t batch_size = input_blob_->getTensorDesc().getDims()[0];
	infer_request_.Infer();

	const SizeVector outputDim = outInfo_.begin()->second->getTensorDesc().getDims();
	int dimention = 512;
	std::vector<float> embedding;
	embedding.reserve(dimention);
	float* output_blob = infer_request_.GetBlob(outInfo_.begin()->first)->buffer().as<float*>();
	std::cout << "Dimention" << outputDim[1] << std::endl;
	for (int i = 0; i < output_dimention; i++) {
		embedding.push_back(output_blob[i]);
	}
	return embedding;
}

void CnnBase::PrintPerformanceCounts(std::string fullDeviceName) const {
	std::cout << "Performance counts for " << config_.path_to_model << std::endl << std::endl;
	::printPerformanceCounts(infer_request_, std::cout, fullDeviceName, false);
}