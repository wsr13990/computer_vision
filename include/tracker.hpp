#ifndef TRACK
#define TRACK

#include "opencv2/tracking.hpp"
#include "core.hpp"


class ObjectTrackers
{
	private:
		int max_tracker;
		std::vector<cv::Ptr<cv::Tracker>> multiTracker;
		TrackedObjects results_;

	public:
		ObjectTrackers(const int &max_tracker);

		void addTracker(const cv::Mat &frame, const TrackedObject &obj);
		void clear();

		TrackedObjects updateTrackedObjects(const cv::Mat &frame, TrackedObjects objects);
};
#endif // !TRACK