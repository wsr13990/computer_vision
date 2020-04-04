#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <ie_plugin_config.hpp>

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <memory>

#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;

namespace {
	template <typename StreamType>
	void SaveDetectionLogToStream(StreamType& stream,
		const DetectionLog& log) {
		for (const auto& entry : log) {
			std::vector<TrackedObject> objects(entry.objects.begin(),
				entry.objects.end());
			std::sort(objects.begin(), objects.end(),
				[](const TrackedObject& a,
					const TrackedObject& b)
				{ return a.object_id < b.object_id; });
			for (const auto& object : objects) {
				auto frame_idx_to_save = entry.frame_idx;
				stream << frame_idx_to_save << ',';
				stream << object.object_id << ','
					<< object.rect.x << ',' << object.rect.y << ','
					<< object.rect.width << ',' << object.rect.height;
				stream << '\n';
			}
		}
	}
}  // anonymous namespace

void PrintDetectionLog(const DetectionLog& log) {
	SaveDetectionLogToStream(std::cout, log);
}

InferenceEngine::Core LoadInferenceEngine(const std::vector<std::string>& devices,
	const std::string& custom_cpu_library,
	const std::string& custom_cldnn_kernels,
	bool should_use_perf_counter) {
	std::set<std::string> loadedDevices;
	InferenceEngine::Core ie;

	for (const auto& device : devices) {
		if (loadedDevices.find(device) != loadedDevices.end()) {
			continue;
		}

		std::cout << "Loading device " << device << std::endl;
		std::cout << ie.GetVersions(device) << std::endl;

		/** Load extensions for the CPU device **/
		if ((device.find("CPU") != std::string::npos)) {
#ifdef WITH_EXTENSIONS
			ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
#endif
			if (!custom_cpu_library.empty()) {
				// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
				auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(custom_cpu_library);
				ie.AddExtension(std::static_pointer_cast<InferenceEngine::IExtension>(extension_ptr), "CPU");
			}
		}
		else if (!custom_cldnn_kernels.empty()) {
			// Load Extensions for other plugins not CPU
			ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, custom_cldnn_kernels} }, "GPU");
		}

		if (device.find("CPU") != std::string::npos) {
			ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES} }, "CPU");
		}
		else if (device.find("GPU") != std::string::npos) {
			ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES} }, "GPU");
		}

		if (should_use_perf_counter)
			ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES} });

		loadedDevices.insert(device);
	}

	return ie;
}

void createAndWriteEmbedding(std::string& photo_reference_dir, std::string& embedding_file,
	CnnBase& facenet, ObjectDetector& detector) {
	std::vector<std::string> file_list = getFileName(photo_reference_dir);
	cv::FileStorage file(embedding_file, cv::FileStorage::WRITE);
	for (int i = 0; i < file_list.size(); i++) {
		std::string fullpath = photo_reference_dir + "/" + file_list[i];
		cv::Mat embedding = facenet.InferFromFile(fullpath, detector, i);

		// Write to file!
		size_t lastindex = file_list[i].find_last_of(".");
		std::string person_name = file_list[i].substr(0, lastindex);
		file << person_name << embedding;
	}
	file.release();
}

//This function update the common name based most frequent name in array
void updateCommonName(cv::Mat& frame, TrackedObjects& tracked_obj,
	CnnBase& facenet, cv::Mat& embedding_reference, float& embedding_treshold,
	std::vector<std::string>& name_list) {
	std::string person_name = "Unidentified";
	cv::Mat distance;
	for (int i = 0; i < tracked_obj.size(); i++) {
		getRoI(frame, tracked_obj);
	}
	for (int i = 0; i < tracked_obj.size();i++) {
		std::cout << "Getting roi" << std::endl;
		cv::Mat roi = tracked_obj[i].roi;
		std::cout << "Getting embedding" << std::endl;
		if (roi.rows > 1) {
			cv::Mat embedding = facenet.Infer(roi);

			//Calculate eucledian distance between embedidng & reference
			for (int i = 0; i < embedding_reference.rows; i++) {
				distance.push_back(cv::norm(embedding, embedding_reference.row(i)));
			}
		}

		//Identify person name
		double min_embedding;
		cv::Point min_loc;
		cv::minMaxLoc(distance, &min_embedding, NULL, &min_loc, NULL);
		if (min_embedding < embedding_treshold) {
			person_name = name_list[min_loc.y];
		}
		else {
			//If most frequent name < treshold we still flag as unidentified
			person_name = "Unidentified";
		}
		tracked_obj[i].names.push_back(person_name);
		tracked_obj[i].getCommonName();
		if (tracked_obj[i].names.size() >= tracked_obj[i].name_limit) {
			tracked_obj[i].names.erase(tracked_obj[i].names.begin());
		}
	}
}