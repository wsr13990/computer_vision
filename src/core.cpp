#include "../include/core.hpp"

#include <iostream>
#include <string>
#include <map>
#include <boost/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void TrackedObject::getRoI(cv::Mat frame) {
	if (rect.x > 0 && rect.y > 0 &&
		(rect.x + rect.width) <= frame.cols &&
		(rect.y + rect.height) <= frame.rows ) {
		roi = frame(rect);
	}
}


bool operator==(const TrackedObject& first, const TrackedObject& second) {
	return ((first.rect == second.rect)
		&& (first.confidence == second.confidence)
		&& (first.frame_idx == second.frame_idx)
		&& (first.object_id == second.object_id)
		&& (first.timestamp == second.timestamp));
}

bool operator!=(const TrackedObject& first, const TrackedObject& second) {
	return !(first == second);
}

void getRoI(cv::Mat &frame, TrackedObjects &obj) {
	for (int i = 0; i < obj.size();i++) {
		obj[i].getRoI(frame);
	}
}

cv::Scalar getRandomColors(){
	cv::RNG rng(cv::getTickCount());
	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	return color;
}

int generateObjectId(TrackedObjects &objects) {
	int maxId = 0;
	for (int i = 0; i < objects.size();i++) {
		int currentId = objects[i].object_id;
		if (currentId > maxId) {
			maxId = currentId;
		}
	}
	return maxId + 1;
}

cv::Mat overlay_tracked_obj(cv::Mat frame, TrackedObjects &tracked_objects, bool showPath, int trajectory_treshold){
	cv::Point2i invalid(-1, -1);
	for (int i = 0; i < tracked_objects.size(); ++i) {
		if (tracked_objects[i].isTracked == true) {
			rectangle(frame, tracked_objects[i].rect, tracked_objects[i].color, 2);

			int x = tracked_objects[i].rect.x;
			int y = tracked_objects[i].rect.y - 5;
			cv::putText(frame,
				tracked_objects[i].common_name,
				cv::Point(x, y), // Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				0.7, // Scale. 2.0 = 2x bigger
				tracked_objects[i].color, // BGR Color
				1); // Line Thickness (Optional)

			if (showPath == true && tracked_objects[i].tracks.size() > 1) {
				for (int j = 0; j < tracked_objects[i].tracks.size()-1; j++) {
					cv::Point2i currPoint = tracked_objects[i].tracks[j + 1];
					cv::Point2i prevPoint = tracked_objects[i].tracks[j];
					double distance = cv::norm(currPoint-prevPoint);
					if (distance < trajectory_treshold) {
						cv::arrowedLine(frame, prevPoint,currPoint,
							tracked_objects[i].color, 1);
					}
				}
			}
		}
	}
	return frame;
}

void removeNonTrackedObj(TrackedObjects obj) {
	for (int i = 0; i < obj.size();i++) {
		if (obj[i].isTracked == false) {
			obj.erase(obj.begin() + i);
		}
	}
}

struct path_leaf_string
{
	std::string operator()(const boost::filesystem::directory_entry& entry) const
	{
		return entry.path().leaf().string();
	}
};

std::vector<std::string> getFileName(const std::string& directory) {
	std::vector<std::string> result;
	boost::filesystem::path p(directory);
	boost::filesystem::directory_iterator start(p);
	boost::filesystem::directory_iterator end;
	std::transform(start, end, std::back_inserter(result), path_leaf_string());
	return result;
}

template<class InputIt, class T = typename std::iterator_traits<InputIt>::value_type>
T most_common(InputIt begin, InputIt end)
{
	std::map<T, int> counts;
	for (InputIt it = begin; it != end; ++it) {
		if (counts.find(*it) != counts.end()) {
			++counts[*it];
		}
		else {
			counts[*it] = 1;
		}
	}
	return std::max_element(counts.begin(), counts.end(),
		[](const std::pair<T, int>& pair1, const std::pair<T, int>& pair2) {
			return pair1.second < pair2.second;})->first;
}

void TrackedObject::getCommonName() {
	common_name = most_common(names.begin(), names.end());
}
