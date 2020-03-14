#ifndef KUHN_MUNKRES
#define KUHN_MUNKRES

#include <map>
#include "opencv2/objdetect.hpp"
#include "core.hpp"

using namespace std;
using namespace cv;

class KuhnMunkres {
public:
	KuhnMunkres();
	void CalculateDissimilarity(TrackedObjects objects1, vector<Rect> obj2);
	map<int, int> getSameObjectsIndex(TrackedObjects obj1, vector<Rect> obj2);
	vector<int> getNewObjects(TrackedObjects obj1, vector<Rect> obj2);
};

#endif // !DETECT