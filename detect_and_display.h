#ifndef DETECT_AND_DISPLAY
#define DETECT_AND_DISPLAY

#include "opencv2/tracking.hpp"

using namespace std;
using namespace cv;


struct TrackerParams {
	size_t min_track_duration;  ///< Min track duration in milliseconds.

	size_t forget_delay;  ///< Forget about track if the last bounding box in
						  /// track was detected more than specified number of
						  /// frames ago.

	float aff_thr_fast;  ///< Affinity threshold which is used to determine if
						 /// tracklet and detection should be combined (fast
						 /// descriptor is used).

	float aff_thr_strong;  ///< Affinity threshold which is used to determine if
						   /// tracklet and detection should be combined(strong
						   /// descriptor is used).

	float shape_affinity_w;  ///< Shape affinity weight.

	float motion_affinity_w;  ///< Motion affinity weight.

	float time_affinity_w;  ///< Time affinity weight.

	float min_det_conf;  ///< Min confidence of detection.

	cv::Vec2f bbox_aspect_ratios_range;  ///< Bounding box aspect ratios range.

	cv::Vec2f bbox_heights_range;  ///< Bounding box heights range.

	int predict;  ///< How many frames are used to predict bounding box in case
	/// of lost track.

	float strong_affinity_thr;  ///< If 'fast' confidence is greater than this
								/// threshold then 'strong' Re-ID approach is
								/// used.

	float reid_thr;  ///< Affinity threshold for re-identification.

	bool drop_forgotten_tracks;  ///< Drop forgotten tracks. If it's enabled it
								 /// disables an ability to get detection log.

	int max_num_objects_in_track;  ///< The number of objects in track is
								   /// restricted by this parameter. If it is negative or zero, the max number of
								   /// objects in track is not restricted.

	///
	/// Default constructor.
	///
	TrackerParams();
};


class FaceDetector
{
	private:
		CascadeClassifier face_cascade;
		int max_tracker;
		bool isTracked;
		vector<Rect> bboxes;
		vector<Scalar> colors;
		vector<Ptr<Tracker>> multiTracker;
		vector<Rect> faces;
		

	public:
		FaceDetector(String &face_cascade_name, const int max_tracker);
		void detectAndDisplay(const Mat frame, int &frame_counter);
		void getRandomColors(const int &numColors);
		void addTracker(const Mat frame, Rect &face);
		void removeTracker(int &tracker_index);
};

#endif // !DETECT_AND_DISPLAY

