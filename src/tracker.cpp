#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include "../include/tracker.hpp"

#include <iostream>


ObjectTrackers::ObjectTrackers(const int &max_track) {
	//We set maximum number of tracker
	max_tracker = max_track;
	multiTracker.reserve(max_tracker);
}

void ObjectTrackers::addTracker(const cv::Mat &frame, const TrackedObject &obj) {
	//Add tracker and bounding box
	cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
	cv::Rect2d obj2d(obj.rect);
	tracker->init(frame, obj2d);
	multiTracker.push_back(tracker);
}

void ObjectTrackers::clear() {
	multiTracker.clear();
}

TrackedObjects ObjectTrackers::updateTrackedObjects(cv::Mat frame, TrackedObjects objects) {
	for (int i = 0; i < multiTracker.size();i++) {
		std::cout << "Multitracker size: "<< multiTracker.size() <<std::endl;
		cv::Rect2d obj2d(objects[i].rect);
		if (!obj2d.empty()) {
			std::cout << "Tracked obj rect: " << obj2d << std::endl; 
			std::cout << "1" << std::endl;
			bool trackingStatus = multiTracker[i]->update(frame, obj2d);
			std::cout << "2" << std::endl;
			objects[i].isTracked = trackingStatus;
			//cout << "Tracking status: " << trackingStatus << endl;
			std::cout << "3" << std::endl;
			cv::Rect obj(obj2d);
			std::cout << "4" << std::endl;
			objects[i].rect = obj;
			std::cout << "5" << std::endl;
			cv::Point2i centerPoint(round(obj.x + (obj.width) / 2),
				round(obj.y + obj.height));
			std::cout << "6" << std::endl;
			objects[i].tracks.push_back(centerPoint);
			std::cout << "7" << std::endl;
		}
	}
	return objects;
}