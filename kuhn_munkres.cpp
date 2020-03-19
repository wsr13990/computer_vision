#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"

//using namespace std;
//using namespace cv;

#include <map>
#include "kuhn_munkres.hpp"

KuhnMunkres::KuhnMunkres(){
}

void KuhnMunkres::CalculateDissimilarity(TrackedObjects obj1, std::vector<cv::Rect> obj2) {
	//TODO: Implement calculating dissimilarity
}

std::map<int, int> KuhnMunkres::getSameObjectsIndex(TrackedObjects obj1, std::vector<cv::Rect> obj2) {
	//TODO: Implement this
}

std::vector<int> KuhnMunkres::getNewObjects(TrackedObjects obj1, std::vector<cv::Rect> obj2) {
	//TODO: Implement this
}

float KuhnMunkres::ShapeAffinity(float weight, const cv::Rect& trk,
	const cv::Rect& det) {
	float w_dist = static_cast<float>(std::abs(trk.width - det.width) / (trk.width + det.width));
	float h_dist = static_cast<float>(std::abs(trk.height - det.height) / (trk.height + det.height));
	return static_cast<float>(exp(static_cast<double>(-weight * (w_dist + h_dist))));
}

float KuhnMunkres::MotionAffinity(float weight, const cv::Rect& trk,
	const cv::Rect& det) {
	float x_dist = static_cast<float>(trk.x - det.x)* (trk.x - det.x) /
		(det.width * det.width);
	float y_dist = static_cast<float>(trk.y - det.y)* (trk.y - det.y) /
		(det.height * det.height);
	return static_cast<float>(exp(static_cast<double>(-weight * (x_dist + y_dist))));
}

float KuhnMunkres::TimeAffinity(float weight, const float& trk_time,
	const float& det_time) {
	return static_cast<float>(exp(static_cast<double>(-weight * std::fabs(trk_time - det_time))));
}

float KuhnMunkres::AffinityFast(const cv::Mat& descriptor1,
	const TrackedObject& obj1,
	const cv::Mat& descriptor2,
	const TrackedObject& obj2) {
	const float eps = 1e-6f;
	float shp_aff = ShapeAffinity(shape_affinity_w, obj1.rect, obj2.rect);
	if (shp_aff < eps) return 0.0f;

	float mot_aff =
		MotionAffinity(motion_affinity_w, obj1.rect, obj2.rect);
	if (mot_aff < eps) return 0.0f;
	float time_aff =
		TimeAffinity(time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));

	if (time_aff < eps) return 0.0f;

	//float app_aff = 1.0f - distance_fast.Compute(descriptor1, descriptor2);

	//return shp_aff * mot_aff * app_aff * time_aff;
	return shp_aff * mot_aff * time_aff;
}

float KuhnMunkres::Affinity(const TrackedObject& obj1,
	const TrackedObject& obj2) {
	float shp_aff = ShapeAffinity(shape_affinity_w, obj1.rect, obj2.rect);
	float mot_aff =
		MotionAffinity(motion_affinity_w, obj1.rect, obj2.rect);
	float time_aff =
		TimeAffinity(time_affinity_w, static_cast<float>(obj1.frame_idx), static_cast<float>(obj2.frame_idx));
	return shp_aff * mot_aff * time_aff;
}

