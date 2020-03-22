#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#include "tracker.hpp"

#include <iostream>

//using namespace std;
//using namespace cv;


ObjectTrackers::ObjectTrackers(const int &max_track) {
	//We set maximum number of tracker
	max_tracker = max_track;
	multiTracker.reserve(max_tracker);
}

void ObjectTrackers::addTracker(const Mat &frame, const TrackedObject &obj) {
	//Add tracker and bounding box
	Ptr<Tracker> tracker = TrackerKCF::create();
	Rect2d obj2d(obj.rect);
	tracker->init(frame, obj2d);
	multiTracker.push_back(tracker);
}

void ObjectTrackers::clear() {
	multiTracker.clear();
}

TrackedObjects ObjectTrackers::updateTrackedObjects(Mat frame, TrackedObjects objects) {
	for (int i = 0; i < multiTracker.size();i++) {
		Rect2d obj2d(objects[i].rect);
		if (&obj2d != NULL) {
			bool trackingStatus = multiTracker[i]->update(frame, obj2d);
			objects[i].isTracked = trackingStatus;
			//cout << "Tracking status: " << trackingStatus << endl;
			Rect obj(obj2d);
			objects[i].rect = obj;
		}
	}
	return objects;
}