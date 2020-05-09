#pragma once

#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

class CustomROI{
protected:
    bool roiEnable=false;
    bool isDrawing=false;
    std::vector<cv::Point> roiShape;
    cv::Scalar color = cv::Scalar(0,255,0);
    cv::Point cursor;
    
public:
    static void createCustomROI(int event, int x, int y, int flags, void* this_){
        CustomROI *self = static_cast<CustomROI*>(this_);
        self->updateCursor(event, x, y);
        self->doCreateCustomROI(event, x, y);
    }
    void doCreateCustomROI(int event, int x, int y);
    void updateCursor(int event, int x, int y);
    void drawCursor(cv::Mat &frame);
    void drawCustomROI(cv::Mat &frame,const std::string &window, std::vector<cv::Point> &roi);
    void processCustomROI(cv::Mat &frame, const std::string &window);
};