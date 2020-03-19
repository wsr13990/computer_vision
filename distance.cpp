#include "distance.hpp"

#include <vector>

CosDistance::CosDistance(const cv::Size& descriptor_size)
	: descriptor_size_(descriptor_size) {
}

float CosDistance::Compute(const cv::Mat& descr1, const cv::Mat& descr2) {
	double xy = descr1.dot(descr2);
	double xx = descr1.dot(descr1);
	double yy = descr2.dot(descr2);
	double norm = sqrt(xx * yy) + 1e-6;
	return 0.5f * static_cast<float>(1.0 - xy / norm);
}

std::vector<float> CosDistance::Compute(const std::vector<cv::Mat>& descrs1,
	const std::vector<cv::Mat>& descrs2) {
	std::vector<float> distances(descrs1.size(), 1.f);
	for (size_t i = 0; i < descrs1.size(); i++) {
		distances.at(i) = Compute(descrs1.at(i), descrs2.at(i));
	}

	return distances;
}

float MatchTemplateDistance::Compute(const cv::Mat& descr1,
	const cv::Mat& descr2) {
	cv::Mat res;
	cv::matchTemplate(descr1, descr2, res, type_);
	float dist = res.at<float>(0, 0);
	return scale_ * dist + offset_;
}

std::vector<float> MatchTemplateDistance::Compute(const std::vector<cv::Mat>& descrs1,
	const std::vector<cv::Mat>& descrs2) {
	std::vector<float> result;
	for (size_t i = 0; i < descrs1.size(); i++) {
		result.push_back(Compute(descrs1[i], descrs2[i]));
	}
	return result;
}