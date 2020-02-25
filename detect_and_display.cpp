#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "detect_and_display.h"
#include <iostream>

using namespace std;
using namespace cv;

FaceDetector::FaceDetector(String face_cascade_name) {
	face_cascade.load(face_cascade_name);
}

void FaceDetector::detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		rectangle(frame, faces[i], (100, 100, 255),2);


		Mat faceROI = frame_gray(faces[i]);
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}