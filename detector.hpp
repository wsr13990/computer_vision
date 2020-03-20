#ifndef DETECT
#define DETECT

#include "opencv2/objdetect.hpp"
#include "core.hpp"
#include "kuhn_munkres.hpp"

using namespace std;
using namespace cv;

class ObjectDetector
{
private:
	CascadeClassifier face_cascade;
	int max_tracker_;
	vector<Rect> bboxes;
	KuhnMunkres solver;
	TrackedObjects results_;

public:
	ObjectDetector(String face_cascade_name, int max_tracker = 10);

	vector<Rect> getBoundingBox(Mat frame);
	TrackedObjects updateTrackedObjects(Mat frame, TrackedObjects objects);
};
#endif // !DETECT

