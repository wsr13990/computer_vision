#ifndef DISTANCE
#define DISTANCE

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <vector>

class CosDistance{
public:
	CosDistance(const cv::Size& descriptor_size);

	float Compute(const cv::Mat& descr1, const cv::Mat& descr2);
	std::vector<float> Compute(
		const std::vector<cv::Mat>& descrs1,
		const std::vector<cv::Mat>& descrs2);

private:
	cv::Size descriptor_size_;
};


class MatchTemplateDistance{
public:
	MatchTemplateDistance(int type = cv::TemplateMatchModes::TM_CCORR_NORMED,
		float scale = -1, float offset = 1)
		: type_(type), scale_(scale), offset_(offset) {}
	float Compute(const cv::Mat& descr1, const cv::Mat& descr2);
	std::vector<float> Compute(const std::vector<cv::Mat>& descrs1,
		const std::vector<cv::Mat>& descrs2);
	virtual ~MatchTemplateDistance() {}

private:
	int type_;      ///< Method of MatchTemplate function computation.
	float scale_;   ///< Scale parameter for the distance. Final distance is
					/// computed as: scale * distance + offset.
	float offset_;  ///< Offset parameter for the distance. Final distance is
					/// computed as: scale * distance + offset.
};
#endif // !DETECT

