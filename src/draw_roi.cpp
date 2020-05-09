#include <opencv2/highgui.hpp>

#include "../include/draw_roi.hpp"

void CustomROI::doCreateCustomROI(int event, int x, int y){
    if ( isDrawing == true && event == cv::EVENT_LBUTTONUP){
        roiShape.push_back(cv::Point(x,y));
    }
}

void CustomROI::updateCursor(int event, int x, int y){
    cursor = cv::Point(x,y);
}

void CustomROI::drawCursor(cv::Mat &frame){
    if ( isDrawing == false){
        return;
    } else {
        cv::drawMarker(frame, cursor,  color, cv::MARKER_CROSS, 15, 1);
    }
}

void CustomROI::drawCustomROI(cv::Mat &frame,const std::string &window, std::vector<cv::Point> &roi){
    if (roiEnable == false || roiShape.size() == 0){
        return;
    }
    cv::Point start_point = roiShape[0];
    
    cv::Point zero_point = start_point;
    for (int i = 0; i < roi.size(); i++){
        std::cout << roi[i] << std::endl;
        cv::Point end_point = roi[i];
        cv::line(frame,start_point, end_point,color,2);
        start_point = end_point;
    }
    cv::line(frame,start_point, zero_point,color,2);
    cv::imshow(window, frame);
}

void CustomROI::processCustomROI(cv::Mat &frame, const std::string &window){
//    Refer to ascii table here to check what this int correspond to
//    http://www.asciitable.com/
    cv::setMouseCallback(window,createCustomROI,this);
    int key = cv::waitKey(10) & 0xFF ;
    if (key == 115){//"s" letter
        isDrawing=!isDrawing;
        roiEnable=isDrawing;
    }
    if (roiEnable == true && key == 13){ //"enter"
        isDrawing=false;
    }
    if (key == 8) { //"delete"
        roiShape.clear();
    }
    drawCursor(frame);
    drawCustomROI(frame,window,roiShape);
}

