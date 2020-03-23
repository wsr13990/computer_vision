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

	TrackedObject()
		: confidence(-1),
		frame_idx(-1),
		object_id(-1),
		isTracked(false),
		timestamp(0) {}

	TrackedObject(const cv::Rect& rect, float confidence, int frame_idx, cv::Scalar color,
		int object_id, bool isTracked)
		: rect(rect),
		confidence(confidence),
		frame_idx(frame_idx),
		object_id(object_id),
		isTracked(isTracked),
		color(color),
		timestamp(0) {}
};

using TrackedObjects = std::deque<TrackedObject>;

cv::Scalar getRandomColors();
int generateObjectId(TrackedObjects& objects);

void display(cv::Mat frame, TrackedObjects &tracked_objects);

void removeNonTrackedObj(TrackedObjects obj);

bool operator==(const TrackedObject& first, const TrackedObject& second);
bool operator!=(const TrackedObject& first, const TrackedObject& second);

using ObjectTracks = std::unordered_map<int, TrackedObjects>;

#endif // !CORE

