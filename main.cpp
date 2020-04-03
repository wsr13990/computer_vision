#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "tracker.hpp"
#include "haar_cascade_detector.hpp"
#include "detector.hpp"
#include "cnn.hpp"

#include <iostream>
#include "core.hpp"

#include <string>
#include <codecvt>
#include <locale>
#include <fstream>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <ext_list.hpp>

#include "utils.hpp"

// TODO:
// Implement logging (what information needed in logging?)

// Implement external model loading (facenet or arcface)
// Implement embedding calculation & faceobject class (should it combined in tracked object or we make separate class?)
// Create face embedding database
// Calculate similarity between embedding
// Create face object identifier by considering indetification from multiple timeframe

int main_work(int argc, const char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();

	cv::String face_cascade_name = cv::samples::findFile(parser.get<cv::String>("face_cascade"));

	int camera_device = parser.get<int>("camera");
	cv::VideoCapture capture;

	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
		std::cout << "--(!)Error opening video capture\n";
		return -1;
	}

	//================================================================================
	//Parameters
	//================================================================================
	int max_tracker = 10;
	cv::Mat frame;

	std::string detector_mode = "CPU";
	std::string reid_mode = "CPU";
	std::string custom_cpu_library = "";
	std::string device = "CPU";
	std::string path_to_custom_layers = "";
	
	bool should_use_perf_counter = false;
	bool recalculate_embedding = true;

	std::string photo_reference_dir = "D:/BELAJAR/OpenVino/facial_recognition/data/photo";
	std::string embedding_file = "D:/BELAJAR/C++/facial_recognition/embedding/vector.txt";
	//================================================================================


	//================================================================================
	//Facenet Model IR
	//================================================================================
	std::string facenet_weight =
		"D:/BELAJAR/C++/facial_recognition/model/ir_facenet/20180408-102900.bin";
	std::string facenet_xml =
		"D:/BELAJAR/C++/facial_recognition/model/ir_facenet/20180408-102900.xml";
	//================================================================================


	//================================================================================
	//Face Detector Model IR
	//================================================================================
	std::string det_weight =
		"D:/BELAJAR/C++/facial_recognition/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin";
	std::string det_xml =
		"D:/BELAJAR/C++/facial_recognition/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
	//================================================================================

	// Load the Infrerence Engine
	std::vector<std::string> devices{ detector_mode, reid_mode };
	InferenceEngine::Core ie =
		LoadInferenceEngine(
			devices, custom_cpu_library, path_to_custom_layers,
			should_use_perf_counter);
	ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), "CPU");
	std::cout << "Load inference engine" << std::endl;

	//Instantiate kuhn munkres detector. For testing purpose, comment out
	//FaceDetector face_detector(face_cascade_name, max_tracker);

	//Instantiate Tracker
	ObjectTrackers tracker(max_tracker);

	//Instantiate Hungarian Algorithm Solver to combine detection & tracking object
	KuhnMunkres solver;

	//Instantiate OpenVino Detector
	DetectorConfig detector_confid(det_xml, det_weight);
	ObjectDetector detector(detector_confid, ie, detector_mode);
	std::cout << "Instantiate detector" << std::endl;

	////Instantiate Facenet
	CnnConfig facenet_config(facenet_xml, facenet_weight);
	CnnBase facenet(facenet_config, ie, detector_mode);
	facenet.Load();

	TrackedObjects objects;
	TrackedObjects tracked_obj;
	TrackedObjects detected_obj;

	int frame_idx = 0;
	int interval = 10;
	double video_fps;
	bool processing = false;
	cv::Mat dissimilarity_mtx;

	//Create reference embedding from photo directory
	if (recalculate_embedding == true) {
		std::vector<std::string> file_list = getFileName(photo_reference_dir);
		for (int i = 0; i < file_list.size(); i++) {
			std::string fullpath = photo_reference_dir + "/" + file_list[i];
			std::vector<float> embedding = facenet.InferFromFile(fullpath, detector);
			//std::cout << embedding << std::endl;
			std::ofstream outFile(embedding_file);
			for (const auto& e : embedding) outFile << e << ",";
			outFile << "\n";
			outFile.close();
		}
	}
	else {

	}

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			std::cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		video_fps = capture.get(cv::CAP_PROP_FPS);
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);

		//For a time interval re-run detection & update tracked object by
		// combine it with tracking bbox using solver
		if (frame_idx % interval == 0) {
			//In each 10 frame, we detect objects and initiate fresh new tracker
			//Thent track it for 10 more frame

			//OpenVino Detector submit & fetch
			detector.submitFrame(frame, frame_idx);
			detector.waitAndFetchResults();
			std::cout << "Submit frame" << std::endl;

			//face_detector.updateTrackedObjects(frame, detected_obj, frame_idx);
			detected_obj = detector.getResults();
			getRoI(frame, detected_obj);

			//Get RoI for each tracked Object
			if (detected_obj.size() > 0) {
				if (processing == true) {
					dissimilarity_mtx = solver.ComputeDissimilarityMatrix(
						tracked_obj, detected_obj);
					std::cout << dissimilarity_mtx << std::endl;
					if (!dissimilarity_mtx.empty()) {
						std::vector<int> result = solver.Solve(dissimilarity_mtx);
						solver.UpdateAndRemoveNonMatch(result, tracked_obj, detected_obj);
						solver.AddNewTrackedObject(result, tracked_obj, detected_obj);
					}
				}
				tracker.clear();
				if (tracked_obj.size() > 0) {
					//If tracked object not null build tracker using that
					for (int i = 0; i < tracked_obj.size();i++) {
						tracker.addTracker(frame, tracked_obj[i]);
					}
				}
				else {
					//Else build tracker using detected object and update processing flag
					for (int i = 0; i < detected_obj.size();i++) {
						tracker.addTracker(frame, detected_obj[i]);
					}
					processing = true;
				}
			}
		}
		//Update tracker
		if (tracked_obj.size() > 0) {
			//If tracked object not null update tracker using that
			tracked_obj = tracker.updateTrackedObjects(frame, tracked_obj);
		}
		else if (detected_obj.size() > 0) {
			//Else update tracker using detected object and update processing flag
			tracked_obj = tracker.updateTrackedObjects(frame, detected_obj);
		}
		else if (detected_obj.size() == 0) {
			processing = false;
		}

		////Create embedding vector using Facenet
		getRoI(frame, tracked_obj);
		for (int i = 0; i < tracked_obj.size();i++) {
			cv::Mat roi = tracked_obj[i].roi;
			std::cout << "Getting embedding" << std::endl;
			std::vector<float> embedding = facenet.Infer(roi);
			//std::cout << "[";
			//for (int i = 0; i < embedding.size(); i++) {
			//	std::cout << embedding[i] << ",";
			//}
			//std::cout << "]"<< std::endl;
		}

		display(frame, tracked_obj);
		std::cout << video_fps << " FPS" << std::endl;

		if (cv::waitKey(10) == 27)
		{
			break; // escape
		}
		frame_idx += 1;
	}
	return 0;
}

int main(int argc, const char** argv) {
	try {
		main_work(argc, argv);
	}
	catch (const std::exception & error) {
		std::cerr << "[ ERROR ] " << error.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
		return 1;
	}

	std::cout << "Execution successful" << std::endl;

	return 0;
}