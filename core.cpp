#include "core.hpp"

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>


void TrackedObject::getRoI(cv::Mat frame) {
	roi = frame(rect);
}


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

void getRoI(cv::Mat &frame, TrackedObjects &obj) {
	for (int i = 0; i < obj.size();i++) {
		obj[i].getRoI(frame);
	}
}

cv::Scalar getRandomColors(){
	cv::RNG rng(cv::getTickCount());
	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
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

struct path_leaf_string
{
	std::string operator()(const boost::filesystem::directory_entry& entry) const
	{
		return entry.path().leaf().string();
	}
};

std::vector<std::string> getFileName(const std::string& directory) {
	std::vector<std::string> result;
	boost::filesystem::path p(directory);
	boost::filesystem::directory_iterator start(p);
	boost::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(result), path_leaf_string());
	return result;
}
