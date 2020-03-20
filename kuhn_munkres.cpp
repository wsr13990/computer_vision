#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"

//using namespace std;
//using namespace cv;

#include <map>
#include <set>
#include <iterator>
#include "kuhn_munkres.hpp"

KuhnMunkres::KuhnMunkres(){
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

	float app_aff = 1.0f - distance_fast.Compute(descriptor1, descriptor2);

	return shp_aff * mot_aff * app_aff * time_aff;
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

cv::Mat KuhnMunkres::ComputeDissimilarityMatrix(
	const TrackedObjects& detections, const TrackedObjects& tracking) {
	cv::Mat dissimilarity_mtx(tracking.size(), detections.size(), CV_32F, cv::Scalar(0));
	for (size_t i = 0; i < tracking.size();i++) {
		for (size_t j = 0; j < detections.size(); j++) {
			dissimilarity_mtx.at<double>(0, 0) = Affinity(detections[i],tracking[i]);
		}
	}
	return dissimilarity_mtx;
}

std::vector<size_t> KuhnMunkres::Solve(const cv::Mat& dissimilarity_matrix) {
	double min_val;
	cv::minMaxLoc(dissimilarity_matrix, &min_val);

	n_ = std::max(dissimilarity_matrix.rows, dissimilarity_matrix.cols);
	dm_ = cv::Mat(n_, n_, CV_32F, cv::Scalar(0));
	marked_ = cv::Mat(n_, n_, CV_8S, cv::Scalar(0));
	points_ = std::vector<cv::Point>(n_ * 2);

	dissimilarity_matrix.copyTo(dm_(
		cv::Rect(0, 0, dissimilarity_matrix.cols, dissimilarity_matrix.rows)));

	is_row_visited_ = std::vector<int>(n_, 0);
	is_col_visited_ = std::vector<int>(n_, 0);

	Run();

	std::vector<size_t> results(dissimilarity_matrix.rows, -1);
	for (int i = 0; i < dissimilarity_matrix.rows; i++) {
		const auto ptr = marked_.ptr<char>(i);
		for (int j = 0; j < dissimilarity_matrix.cols; j++) {
			if (ptr[j] == kStar) {
				results[i] = j;
			}
		}
	}
	return results;
}

void KuhnMunkres::TrySimpleCase() {
	auto is_row_visited = std::vector<int>(n_, 0);
	auto is_col_visited = std::vector<int>(n_, 0);

	for (int row = 0; row < n_; row++) {
		auto ptr = dm_.ptr<float>(row);
		auto marked_ptr = marked_.ptr<char>(row);
		auto min_val = *std::min_element(ptr, ptr + n_);
		for (int col = 0; col < n_; col++) {
			ptr[col] -= min_val;
			if (ptr[col] == 0 && !is_col_visited[col] && !is_row_visited[row]) {
				marked_ptr[col] = kStar;
				is_col_visited[col] = 1;
				is_row_visited[row] = 1;
			}
		}
	}
}

bool KuhnMunkres::CheckIfOptimumIsFound() {
	int count = 0;
	for (int i = 0; i < n_; i++) {
		const auto marked_ptr = marked_.ptr<char>(i);
		for (int j = 0; j < n_; j++) {
			if (marked_ptr[j] == kStar) {
				is_col_visited_[j] = 1;
				count++;
			}
		}
	}

	return count >= n_;
}

cv::Point KuhnMunkres::FindUncoveredMinValPos() {
	auto min_val = std::numeric_limits<float>::max();
	cv::Point min_val_pos(-1, -1);
	for (int i = 0; i < n_; i++) {
		if (!is_row_visited_[i]) {
			auto dm_ptr = dm_.ptr<float>(i);
			for (int j = 0; j < n_; j++) {
				if (!is_col_visited_[j] && dm_ptr[j] < min_val) {
					min_val = dm_ptr[j];
					min_val_pos = cv::Point(j, i);
				}
			}
		}
	}
	return min_val_pos;
}

void KuhnMunkres::UpdateDissimilarityMatrix(float val) {
	for (int i = 0; i < n_; i++) {
		auto dm_ptr = dm_.ptr<float>(i);
		for (int j = 0; j < n_; j++) {
			if (is_row_visited_[i]) dm_ptr[j] += val;
			if (!is_col_visited_[j]) dm_ptr[j] -= val;
		}
	}
}

int KuhnMunkres::FindInRow(int row, int what) {
	for (int j = 0; j < n_; j++) {
		if (marked_.at<char>(row, j) == what) {
			return j;
		}
	}
	return -1;
}

int KuhnMunkres::FindInCol(int col, int what) {
	for (int i = 0; i < n_; i++) {
		if (marked_.at<char>(i, col) == what) {
			return i;
		}
	}
	return -1;
}

void KuhnMunkres::Run() {
	TrySimpleCase();
	while (!CheckIfOptimumIsFound()) {
		while (true) {
			auto point = FindUncoveredMinValPos();
			auto min_val = dm_.at<float>(point.y, point.x);
			if (min_val > 0) {
				UpdateDissimilarityMatrix(min_val);
			}
			else {
				marked_.at<char>(point.y, point.x) = kPrime;
				int col = FindInRow(point.y, kStar);
				if (col >= 0) {
					is_row_visited_[point.y] = 1;
					is_col_visited_[col] = 0;
				}
				else {
					int count = 0;
					points_[count] = point;

					while (true) {
						int row = FindInCol(points_[count].x, kStar);
						if (row >= 0) {
							count++;
							points_[count] = cv::Point(points_[count - 1].x, row);
							int col = FindInRow(points_[count].y, kPrime);
							count++;
							points_[count] = cv::Point(col, points_[count - 1].y);
						}
						else {
							break;
						}
					}

					for (int i = 0; i < count + 1; i++) {
						auto& mark = marked_.at<char>(points_[i].y, points_[i].x);
						mark = mark == kStar ? 0 : kStar;
					}

					is_row_visited_ = std::vector<int>(n_, 0);
					is_col_visited_ = std::vector<int>(n_, 0);

					marked_.setTo(0, marked_ == kPrime);
					break;
				}
			}
		}
	}
}
