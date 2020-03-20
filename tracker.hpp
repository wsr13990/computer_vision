#ifndef TRACK
#define TRACK

#include "opencv2/tracking.hpp"
#include "core.hpp"

using namespace std;
using namespace cv;


class ObjectTrackers
{
	private:
		int max_tracker;
		vector<Ptr<Tracker>> multiTracker;
		TrackedObjects results_;

	public:
		ObjectTrackers(const int max_tracker);

		void addTracker(const Mat frame, const TrackedObject obj);
		void removeTracker(int &tracker_index);
		void clear();

		TrackedObjects updateTrackedObjects(Mat frame, TrackedObjects objects);
};
#endif // !TRACK

