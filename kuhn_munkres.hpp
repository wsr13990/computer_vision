#ifndef KUHN_MUNKRES
#define KUHN_MUNKRES

#include <map>
#include "opencv2/objdetect.hpp"
#include "core.hpp"
#include "distance.hpp"

class KuhnMunkres {
public:
	CosDistance distance_fast = CosDistance(cv::Size(16, 32));
	float shape_affinity_w = 0.5f;
	float motion_affinity_w = 0.2f;
	float time_affinity_w = 0.0f;


	KuhnMunkres();

	float ShapeAffinity(float weight, const cv::Rect& trk,
		const cv::Rect& det);
	float MotionAffinity(float weight, const cv::Rect& trk,
		const cv::Rect& det);
	float TimeAffinity(float weight, const float& trk_time,
		const float& det_time);

	float AffinityFast(const cv::Mat& descriptor1,
		const TrackedObject& obj1,
		const cv::Mat& descriptor2,
		const TrackedObject& obj2);
	float Affinity(const TrackedObject& obj1,
		const TrackedObject& obj2);

	void CalculateDissimilarity(TrackedObjects objects1, std::vector<cv::Rect> obj2);
	std::map<int, int> getSameObjectsIndex(TrackedObjects obj1, std::vector<cv::Rect> obj2);
	std::vector<int> getNewObjects(TrackedObjects obj1, std::vector<cv::Rect> obj2);
};

#endif // !DETECT