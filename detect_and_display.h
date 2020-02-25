#ifndef DETECT_AND_DISPLAY
#define DETECT_AND_DISPLAY

using namespace std;
using namespace cv;

class FaceDetector
{
	private:
		CascadeClassifier face_cascade;

	public:
		FaceDetector(String);
		void detectAndDisplay(Mat frame);
};

#endif // !DETECT_AND_DISPLAY

