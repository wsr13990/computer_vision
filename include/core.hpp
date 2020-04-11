#ifndef CORE
#define CORE

#include <opencv2/core.hpp>

#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>

struct TrackedObject {
	cv::Rect rect;       ///< Detected object ROI (zero area if N/A).
	double confidence;   ///< Detection confidence level (-1 if N/A).
	int frame_idx;       ///< Frame index where object was detected (-1 if N/A).
	int object_id;       ///< Unique object identifier (-1 if N/A).
	uint64_t timestamp;  ///< Timestamp in milliseconds.
	cv::Scalar color; // Colour for the bounding box
	bool isTracked; //Wether actively being tracked
	cv::Mat roi;
	std::vector<std::string> names;
	std::string common_name;
	std::vector<cv::Point2i> tracks;
	int name_limit;
	int name_treshold;
	int label;

	TrackedObject()
		: confidence(-1),
		frame_idx(-1),
		object_id(-1),
		isTracked(false),
		name_treshold(10), //Trehshold minimum number of same name to identify person
		name_limit(20), //Limit for vector person name
		timestamp(0) {
		names.reserve(name_limit);
	}

	TrackedObject(const cv::Rect& rect, float confidence, int frame_idx, cv::Scalar color,
		int object_id, bool isTracked)
		: rect(rect),
		confidence(confidence),
		frame_idx(frame_idx),
		object_id(object_id),
		isTracked(isTracked),
		color(color),
		timestamp(0) {}

	void getRoI(cv::Mat frame);
	void getCommonName();
};

using TrackedObjects = std::deque<TrackedObject>;

cv::Scalar getRandomColors();
int generateObjectId(TrackedObjects& objects);

void getRoI(cv::Mat& frame, TrackedObjects& obj);
cv::Mat display(cv::Mat frame, TrackedObjects &tracked_objects,
	bool showPath = false, int trajectory_treshold = 20);

void removeNonTrackedObj(TrackedObjects obj);

bool operator==(const TrackedObject& first, const TrackedObject& second);
bool operator!=(const TrackedObject& first, const TrackedObject& second);

std::vector<std::string> getFileName(const std::string& directory);

using ObjectTracks = std::unordered_map<int, TrackedObjects>;

#endif // !CORE

