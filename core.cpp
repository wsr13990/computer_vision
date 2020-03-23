#include "core.hpp"

#include <iostream>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>

bool operator==(const TrackedObject& first, const TrackedObject& second) {
	return ((first.rect == second.rect)
		&& (first.confidence == second.confidence)
		&& (first.frame_idx == second.frame_idx)
		&& (first.object_id == second.object_id)
		&& (first.timestamp == second.timestamp));
}

bool operator!=(const TrackedObject& first, const TrackedObject& second) {
	return !(first == second);
}

cv::Scalar getRandomColors(){
	cv::RNG rng(cv::getTickCount());
	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//cout << color;
	return color;
}

int generateObjectId(TrackedObjects &objects) {
	int maxId = 0;
	for (int i = 0; i < objects.size();i++) {
		int currentId = objects[i].object_id;
		if (currentId > maxId) {
			maxId = currentId;
		}
	}
	return maxId + 1;
}

void display(cv::Mat frame, TrackedObjects &tracked_objects){
	for (int i = 0; i < tracked_objects.size(); ++i) {
		if (tracked_objects[i].isTracked == true) {
			rectangle(frame, tracked_objects[i].rect, tracked_objects[i].color, 2);
		}
	}
	imshow("Capture - Face detection", frame);
}

void removeNonTrackedObj(TrackedObjects obj) {
	for (int i = 0; i < obj.size();i++) {
		if (obj[i].isTracked == false) {
			obj.erase(obj.begin() + i);
		}
	}
}
