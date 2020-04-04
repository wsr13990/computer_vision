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
#include "mode.hpp"

// TODO:
// Implement logging (what information needed in logging?)

// Implement processing from video file
// Implement Pedestrian detector

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

	// Input Parameter
	std::string video_file = "D:/BELAJAR/C++/facial_recognition/sample_video/car.mp4";

	// Input Channel Mode
	int input_mode = FILE_VIDEO_INPUT;


	cv::VideoCapture capture;
	if (input_mode == FILE_VIDEO_INPUT) {
	capture.open(video_file);
	}
	else if (input_mode == CAMERA_INPUT) {
		//-- 2. Read the video stream
		capture.open(camera_device);
		int frame_limit = capture.get(cv::CAP_PROP_FRAME_COUNT);
	}

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
	bool recalculate_embedding = false;
	float embedding_treshold = 1.1;

	// Detection Mode
	int mode = PEDESTRIAN_DETECTION;

	std::string photo_reference_dir = "D:/BELAJAR/OpenVino/facial_recognition/data/photo";
	std::string embedding_file = "D:/BELAJAR/C++/facial_recognition/embedding/vector.xml";
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
	std::string det_weight;
	std::string det_xml;
	if (mode == FACIAL_RECOGNITION) {
		det_weight =
			"D:/BELAJAR/C++/facial_recognition/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin";
		det_xml =
			"D:/BELAJAR/C++/facial_recognition/model/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
	}
	else if (mode == PEDESTRIAN_DETECTION) {
		det_weight =
			"D:/BELAJAR/C++/facial_recognition/model/intel/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.bin";
		det_xml =
			"D:/BELAJAR/C++/facial_recognition/model/intel/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.xml";
	}
	//================================================================================

	// Load the Infrerence Engine
	std::vector<std::string> devices{ detector_mode, reid_mode };
	InferenceEngine::Core ie =
		LoadInferenceEngine(
			devices, custom_cpu_library, path_to_custom_layers,
			should_use_perf_counter);
	ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), "CPU");
	std::cout << "Load inference engine" << std::endl;

	std::cout << "Instantiate detector" << std::endl;

	TrackedObjects tracked_obj;

	//================================================================================
	//Instantiate Facenet (Face Recognition)
	//================================================================================
	CnnConfig facenet_config(facenet_xml, facenet_weight);
	CnnBase facenet(facenet_config, ie, detector_mode);
	facenet.Load();
	//================================================================================
	
	//================================================================================
	//Instantiate Hungarian Algorithm Solver to combine detection & tracking object
	KuhnMunkres solver;
	//================================================================================

	//================================================================================
	//Instantiate Tracker & Detector & Object Detected
	//================================================================================
	ObjectTrackers tracker(max_tracker);
	DetectorConfig detector_confid(det_xml, det_weight);
	std::cout << det_weight << std::endl << std::endl;
	std::cout << det_xml << std::endl << std::endl;
	ObjectDetector detector(detector_confid, ie, detector_mode);
	//================================================================================

	TrackedObjects objects;
	TrackedObjects detected_obj;

	int frame_idx = 0;
	int interval = 5;
	double video_fps;
	bool processing = false;
	cv::Mat dissimilarity_mtx;

	//Create reference embedding from photo directory
	if (recalculate_embedding == true && mode == FACIAL_RECOGNITION) {
		createAndWriteEmbedding(photo_reference_dir, embedding_file, facenet, detector);
	}

	//Read person name and it's embedding from file
	cv::Mat embedding_reference;
	std::vector<std::string> name_list;
	cv::FileStorage file(embedding_file, cv::FileStorage::READ);
	cv::FileNode fn = file.root();
	for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit) {
		cv::FileNode item = *fit;
		name_list.push_back(item.name());
		cv::Mat row;
		item >> row;
		embedding_reference.push_back(row);
	}

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			std::cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		//================================================================================
		//If input come from video, loop the video
		//================================================================================
		int current_frame = capture.get(cv::CAP_PROP_POS_FRAMES);
		int frame_limit = capture.get(cv::CAP_PROP_FRAME_COUNT);
		if (input_mode == FILE_VIDEO_INPUT && current_frame >= frame_limit) {
			capture.set(cv::CAP_PROP_POS_FRAMES, 0);
		}
		//================================================================================

		video_fps = capture.get(cv::CAP_PROP_FPS);
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);

		//For a time interval re-run detection & update tracked object by
		// combine it with tracking bbox using solver
		if (frame_idx % interval == 0) {
			//In each 10 frame, we detect objects and initiate fresh new tracker
			//Thent track it for 10 more frame

			//OpenVino Detector submit & fetch
			std::cout << "Begin" << std::endl;
			cv::Mat frame_sumbitted;
			cv::cvtColor(frame, frame_sumbitted, cv::COLOR_BGR2RGB);

			detector.submitFrame(frame_sumbitted, frame_idx);
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
		if (tracked_obj.size() > 0 && processing == true) {
			//If tracked object not null update tracker using that
			tracked_obj = tracker.updateTrackedObjects(frame, tracked_obj);
		}
		else if (detected_obj.size() > 0) {
			//Else update tracker using detected object and update processing flag
			tracked_obj = tracker.updateTrackedObjects(frame, detected_obj);
			processing = true;
		}
		else if (detected_obj.size() == 0) {
			processing = false;
		}

		//Create embedding vector using Facenet
		if (mode == FACIAL_RECOGNITION && processing == true) {
			updateCommonName(frame, tracked_obj, facenet, embedding_reference,
				embedding_treshold, name_list);
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