#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;

#include <map>
#include "kuhn_munkres.h"

KuhnMunkres::KuhnMunkres(){
	vector<vector<double>> dissimilarity_mtx;
}

void KuhnMunkres::CalculateDissimilarity(TrackedObjects obj1, vector<Rect> obj2) {
	//TODO: Implement calculating dissimilarity
}

map<int, int> KuhnMunkres::getSameObjectsIndex(TrackedObjects obj1, vector<Rect> obj2) {
	//TODO: Implement this
}

vector<int> KuhnMunkres::getNewObjects(TrackedObjects obj1, vector<Rect> obj2) {
	//TODO: Implement this
}

