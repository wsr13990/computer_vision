#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "../include/detect_and_display.h"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include <iostream>

using namespace std;
using namespace cv;

TrackerParams::TrackerParams()
	: min_track_duration(1000),
	forget_delay(150),
	aff_thr_fast(0.8f),
	aff_thr_strong(0.75f),
	shape_affinity_w(0.5f),
	motion_affinity_w(0.2f),
	time_affinity_w(0.0f),
	min_det_conf(0.65f),
	bbox_aspect_ratios_range(0.666f, 5.0f),
	bbox_heights_range(40, 1000),
	predict(25),
	strong_affinity_thr(0.2805f),
	reid_thr(0.61f),
	drop_forgotten_tracks(true),
	max_num_objects_in_track(300) {}


FaceDetector::FaceDetector(String &face_cascade_name, int max_track) {
	//We set maximum number of tracker
	isTracked = 0;
	max_tracker = max_track;
	multiTracker.reserve(max_tracker);
	colors.reserve(max_tracker);
	face_cascade.load(face_cascade_name);
	getRandomColors(max_tracker);
}

void FaceDetector::getRandomColors(const int &numColors)
{
	RNG rng(0);
	for (int i = 0; i < numColors; i++) {
		colors.push_back(Scalar(rng.uniform(0, 255),
			rng.uniform(0, 255), rng.uniform(0, 255)));
	}
}

void FaceDetector::addTracker(const Mat frame,Rect &face) {
	//Add tracker and bounding box
	Ptr<Tracker> tracker = TrackerKCF::create();
	Rect2d face2d(face);
	tracker->init(frame, face2d);
	multiTracker.push_back(tracker);
}

void FaceDetector::removeTracker(int &tracker_index) {
	//Remove tracker and bounding box at specified index
	multiTracker.erase(multiTracker.begin() + tracker_index);
	faces.erase(faces.begin() + tracker_index);
}

void FaceDetector::detectAndDisplay(const Mat frame, int &frame_counter)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	if (faces.size()==0) {
		face_cascade.detectMultiScale(frame_gray, faces);
		getRandomColors(faces.size());
		if (faces.size() > 0) {
			for (int i = 0; i < faces.size(); i++)
			{
				cout << "Detecting\n";
				rectangle(frame, faces[i], colors[i], 2);
				addTracker(frame, faces[i]);
		//		//Mat faceROI = frame_gray(faces[i]);
			}
		}
	}
	else {
		for (int i = 0; i < faces.size(); i++)
		{
			if (faces.size() > 0) {
				Rect2d face(faces[i]);
				isTracked = multiTracker[i]->update(frame, face);
				faces[i] = face;
				cout << "Tracking";
				if (isTracked==1) {
					cout << "Tracking\n";
					rectangle(frame, faces[i], colors[i], 2);
					//Mat faceROI = frame_gray(r);
				}
				else {
					removeTracker(i);
					//cout << frame_counter << " No face detected\n";
				}
			}
		}
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}