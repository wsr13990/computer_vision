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
	map<int, int> same_index = solver.getSameObjectsIndex(objects,bboxes);
	for (std::map<int, int>::value_type& x : same_index) {
		TrackedObject object;
		objects[x.first].rect = bboxes[x.second];
		if (&objects[x.first].color == NULL) {
			objects[x.first].color = getRandomColors();
		}
	}
	vector<int> new_index = solver.getNewObjects(objects, bboxes);
	for (int i = 0; i < new_index.size();i++){
		TrackedObject object;
		object.object_id = generateObjectId(objects);
		object.rect = bboxes[new_index[i]];
		objects.push_back(object);
	}
	return objects;
}