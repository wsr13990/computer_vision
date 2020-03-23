#ifndef DETECT
#define DETECT

#include "opencv2/objdetect.hpp"
#include "core.hpp"
#include "kuhn_munkres.hpp"

class ObjectDetector
{
private:
	cv::CascadeClassifier face_cascade;
	int max_tracker_;
	std::vector<cv::Rect> bboxes;
	KuhnMunkres solver;
	TrackedObjects results_;

public:
	ObjectDetector(const cv::String &face_cascade_name, const int &max_tracker = 10);

	std::vector<cv::Rect> getBoundingBox(cv::Mat &frame);
	void updateTrackedObjects(cv::Mat &frame, TrackedObjects &objects, int &frame_idx);
};
#endif // !DETECT

