#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "tracker.hpp"
#include "detector.hpp"

#include <iostream>
#include "core.hpp"

//using namespace std;
//using namespace cv;

// TODO:
// Implement remove untracked object
// Fix kuhn munkres solver from crashing when one of the object is missing
// Implement logging

// Implement Openvino detection model
// Implement external model loading (facenet or arcface)
// Implement embedding calculation & faceobject class (should it combined in tracked object or we make separate class?)
// Create face embedding database
// Calculate similarity between embedding
// Create face object identifier by considering indetification from multiple timeframe


int main_work(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();

	String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));

	int camera_device = parser.get<int>("camera");
	VideoCapture capture;

	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	int max_tracker = 10;
	Mat frame;
	ObjectDetector detector(face_cascade_name, max_tracker);
	ObjectTrackers tracker(max_tracker);
	KuhnMunkres solver;
	TrackedObjects objects;
	TrackedObjects tracked_obj;
	TrackedObjects detected_obj;
	int frame_idx = 0;
	int interval = 10;
	double video_fps;
	bool processing = false;
	cv::Mat dissimilarity_mtx;

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		video_fps = capture.get(CAP_PROP_FPS);
		//cv::Mat frame = pair.first;
		//int frame_idx = pair.second;
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);
		//cout << cur_timestamp << "\n";

		//-- For a time interval rerun detection & update tracked object by
		// combine it with tracking bbox
		if (frame_idx % interval == 0) {
			//Temporary step before implementing kuhn munkres
			//In each 10 frame, we detect objects and initiate fresh new tracker
			//Thent track it for 10 more frame
			detected_obj = detector.updateTrackedObjects(frame, detected_obj);
			if (detected_obj.size() > 0) {
				if (processing == true) {
					dissimilarity_mtx = solver.ComputeDissimilarityMatrix(
						tracked_obj, detected_obj);
					//cout << dissimilarity_mtx << endl;
					if (!dissimilarity_mtx.empty()) {
						//Result element i with value j mean that tracking index i similar
						//with detection object on index j
						vector<int> result = solver.Solve(dissimilarity_mtx);
						//cout << "Result : ";
						//for (int i = 0; i < result.size();i++) {
						//	cout << result[i] << ",";
						//}
						//cout << endl;
						//cout << "----------------------------------" << endl;

						for (int tracking_idx = result.size()-1; tracking_idx >= 0;
							tracking_idx--) {
							int detection_idx = result[tracking_idx];
							if (detection_idx != -1) {
								//Based on kuhn munkres, update tracked objec bbox & frame idx
								//cout << "Detection index: " << detection_idx << endl;
								tracked_obj[tracking_idx].rect = detected_obj[detection_idx].rect;
								tracked_obj[tracking_idx].frame_idx = frame_idx;
							} else {
								//Update tracking flag for non matched object to false
								tracked_obj.erase(tracked_obj.begin() + tracking_idx);
							}
						}
						//Add new tracked objects
						//Check if there are item in detection object not exists in result
						for (int detection_idx = 0; detection_idx < detected_obj.size();
							detection_idx++) {
							if (std::find(result.begin(), result.end(),
								detection_idx) == result.end()) {
								tracked_obj.push_back(detected_obj[detection_idx]);
							}
						}
					}
				}
				tracker.clear();
				if (processing == true) {
					for (int i = 0; i < tracked_obj.size();i++) {
						tracker.addTracker(frame, tracked_obj[i]);
					}
				}
				else {
					for (int i = 0; i < detected_obj.size();i++) {
						tracker.addTracker(frame, detected_obj[i]);
					}
				}
			}
		}
		//Update tracker
		if (processing == true && tracked_obj.size() > 0) {
			tracked_obj = tracker.updateTrackedObjects(frame, tracked_obj);
		} else if (detected_obj.size() > 0){
			tracked_obj = tracker.updateTrackedObjects(frame, detected_obj);
			processing = true;
		} else if (detected_obj.size() == 0) {
			processing = false;
		}

		display(frame, tracked_obj);

		if (waitKey(10) == 27)
		{
			break; // escape
		}
		frame_idx +=1;
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