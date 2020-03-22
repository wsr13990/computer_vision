#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include "detector.hpp"

#include <iostream>

//using namespace std;
//using namespace cv;


ObjectDetector::ObjectDetector(String face_cascade_name,int max_tracker) {
	//We set maximum number of tracker
	max_tracker_ = max_tracker;
	face_cascade.load(face_cascade_name);
	KuhnMunkres solver();
}

vector<Rect> ObjectDetector::getBoundingBox(Mat frame) {
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, bboxes);
	return bboxes;
}

TrackedObjects ObjectDetector::updateTrackedObjects(Mat frame,TrackedObjects objects) {
	vector<Rect> bboxes;
	bboxes = getBoundingBox(frame);
	//TODO: use kuhn munkres to see what bbox identical to which tracked object rect
	objects.clear();
	for (int i = 0; i < bboxes.size();i++) {
		TrackedObject object;
		object.rect = bboxes[i];
		object.color = getRandomColors();
		objects.push_back(object);
	}
	return objects;
}