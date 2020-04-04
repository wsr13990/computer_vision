#ifndef CNN
#define CNN

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "ocv_common.hpp"
#include "detector.hpp"
#include "cnn_config.hpp"

#include <inference_engine.hpp>

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
	cv::Mat Infer(const cv::Mat& frame) const;
	cv::Mat InferFromFile(std::string& filepath, ObjectDetector& detector, int& index) const;
	cv::Mat Preprocess(std::string& filepath) const;

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
	InferenceEngine::InferRequest::Ptr request;
};

#endif // !CNN