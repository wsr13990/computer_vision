#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "tracker.h"
#include "detector.h"

#include <iostream>
#include "core.hpp"

using namespace std;
using namespace cv;

// TODO:
// Implement logging
// Separate detection, tracking & display
// Implement affinity & dissimilarity matrix
// Implement kuhn munkres solver
// Combine detection & tracking

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
	TrackedObjects objects;
	int frame_counter = 0;
	int interval = 10;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		//-- For a time interval rerun detection & update tracked object by
		// combine it with tracking bbox
		if (frame_counter % interval == 0) {
			//Temporary step before implementing kuhn munkres
			//In each 10 frame, we detect objects and initiate fresh new tracker
			//Thent track it for 10 more frame
			tracker.clear();
			objects = detector.updateTrackedObjects(frame, objects);
			for (int i = 0; i < objects.size();i++) {
				tracker.addTracker(frame, objects[i]);
				cout << "Add tracker \n";
			}
			cout << "Detecting \n";
		}
		objects = tracker.updateTrackedObjects(frame, objects);
		display(frame,objects);

		if (waitKey(10) == 27)
		{
			break; // escape
		}
		frame_counter +=1;
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