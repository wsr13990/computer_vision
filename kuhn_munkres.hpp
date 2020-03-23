#ifndef KUHN_MUNKRES
#define KUHN_MUNKRES

#include <map>
#include <set>
#include <iterator>

#include "opencv2/objdetect.hpp"
#include "core.hpp"
#include "distance.hpp"

class KuhnMunkres {
private:
	static constexpr int kStar = 1;
	static constexpr int kPrime = 2;

	cv::Mat dm_;
	cv::Mat marked_;
	std::vector<cv::Point> points_;

	std::vector<int> is_row_visited_;
	std::vector<int> is_col_visited_;

	int n_;

	void TrySimpleCase();
	bool CheckIfOptimumIsFound();
	cv::Point FindUncoveredMinValPos();
	void UpdateDissimilarityMatrix(float val);
	int FindInRow(int row, int what);
	int FindInCol(int col, int what);
	void Run();

public:
	CosDistance distance_fast = CosDistance(cv::Size(16, 32));
	float shape_affinity_w = 0.0f;
	float motion_affinity_w = 0.9f;
	float time_affinity_w = 0.0f;


	KuhnMunkres();
	void AddNewTrackedObject(std::vector<int>& result, TrackedObjects& tracked_obj,
		TrackedObjects& detected_obj);
	void UpdateAndRemoveNonMatch(std::vector<int>& result, TrackedObjects& tracked_obj,
		TrackedObjects& detected_obj);

	float ShapeAffinity(float weight, const cv::Rect& trk,
		const cv::Rect& det);
	float MotionAffinity(float weight, const cv::Rect& trk,
		const cv::Rect& det);
	float TimeAffinity(float weight, const float& trk_time,
		const float& det_time);
	float Affinity(const TrackedObject& obj1,
		const TrackedObject& obj2);
	float ComputeIoU(const TrackedObject& detection, const TrackedObject& tracking);

	cv::Mat ComputeDissimilarityMatrix(const TrackedObjects& tracking,
		const TrackedObjects& detection,const bool &useIoU = false);
	
	std::vector<int> Solve(const cv::Mat& dissimilarity_matrix);
};

#endif // !KUHN_MUNKRES