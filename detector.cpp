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


ObjectDetector::ObjectDetector(const String &face_cascade_name,const int &max_tracker) {
	//We set maximum number of tracker
	max_tracker_ = max_tracker;
	face_cascade.load(face_cascade_name);
	KuhnMunkres solver();
}

vector<Rect> ObjectDetector::getBoundingBox(Mat &frame) {
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, bboxes);
	return bboxes;
}

void ObjectDetector::updateTrackedObjects(Mat &frame,TrackedObjects &objects, int &frame_idx) {
	vector<Rect> bboxes;
	bboxes = getBoundingBox(frame);
	objects.clear();
	for (int i = 0; i < bboxes.size();i++) {
		TrackedObject object;
		object.rect = bboxes[i];
		object.color = getRandomColors();
		object.frame_idx = frame_idx;
		objects.push_back(object);
	}
}