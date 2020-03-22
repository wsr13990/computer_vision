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
// Implement logging (what information needed in logging?)

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
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);

		//For a time interval rerun detection & update tracked object by
		// combine it with tracking bbox using solver
		if (frame_idx % interval == 0) {
			//In each 10 frame, we detect objects and initiate fresh new tracker
			//Thent track it for 10 more frame
			detector.updateTrackedObjects(frame, detected_obj, frame_idx);
			if (detected_obj.size() > 0) {
				if (processing == true) {
					dissimilarity_mtx = solver.ComputeDissimilarityMatrix(
						tracked_obj, detected_obj);
					std::cout << dissimilarity_mtx << endl;
					if (!dissimilarity_mtx.empty()) {
						vector<int> result = solver.Solve(dissimilarity_mtx);
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
		} else if (detected_obj.size() > 0){
			//Else update tracker using detected object and update processing flag
			tracked_obj = tracker.updateTrackedObjects(frame, detected_obj);
		} else if (detected_obj.size() == 0) {
			processing = false;
		}

		display(frame, tracked_obj);
		cout << video_fps << " FPS"<< endl;

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