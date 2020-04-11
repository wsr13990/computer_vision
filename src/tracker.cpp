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
		cv::Rect2d obj2d(objects[i].rect);
		if (&obj2d != NULL) {
			bool trackingStatus = multiTracker[i]->update(frame, obj2d);
			objects[i].isTracked = trackingStatus;
			//cout << "Tracking status: " << trackingStatus << endl;
			cv::Rect obj(obj2d);
			objects[i].rect = obj;

			cv::Point2i centerPoint(round(obj.x + (obj.width) / 2),
				round(obj.y + obj.height));

			objects[i].tracks.push_back(centerPoint);

		}
	}
	return objects;
}