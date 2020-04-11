#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include "../include/haar_cascade_detector.hpp"

#include <iostream>

FaceDetector::FaceDetector(const cv::String &face_cascade_name,const int &max_tracker) {
	//We set maximum number of tracker
	max_tracker_ = max_tracker;
	face_cascade.load(face_cascade_name);
	KuhnMunkres solver();
}

std::vector<cv::Rect> FaceDetector::getBoundingBox(cv::Mat &frame) {
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, bboxes);
	return bboxes;
}

void FaceDetector::updateTrackedObjects(cv::Mat &frame,TrackedObjects &objects, int &frame_idx) {
	std::vector<cv::Rect> bboxes;
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