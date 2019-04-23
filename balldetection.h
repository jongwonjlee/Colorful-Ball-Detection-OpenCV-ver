#ifndef BALLDETECTION_H
#define BALLDETECTION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
using namespace std;
using namespace cv;


/* Declaration of functions that changes data types */
string intToString(int n);
string floatToString(float f);
void morphOps(Mat &thresh); // Declaration of functions that calculates the ball position from pixel position
vector<float> pixel2point(Point center, int radius);
string type2str(int type);
void remove_trashval(vector<Point2f> &center, vector<float> &radius);

/* trackbar part */
void on_low_h_thresh_trackbar_red(int, void *);
void on_high_h_thresh_trackbar_red(int, void *);
void on_low_h2_thresh_trackbar_red(int, void *);
void on_high_h2_thresh_trackbar_red(int, void *);
void on_low_s_thresh_trackbar_red(int, void *);
void on_high_s_thresh_trackbar_red(int, void *);
void on_low_v_thresh_trackbar_red(int, void *);
void on_high_v_thresh_trackbar_red(int, void *);
int low_h_r=0, high_h_r=6, low_h2_r=167, high_h2_r=180;
int low_s_r=90, low_v_r=102;
int high_s_r=255, high_v_r=255;

void on_low_h_thresh_trackbar_blue(int, void *);
void on_high_h_thresh_trackbar_blue(int, void *);
void on_low_s_thresh_trackbar_blue(int, void *);
void on_high_s_thresh_trackbar_blue(int, void *);
void on_low_v_thresh_trackbar_blue(int, void *);
void on_high_v_thresh_trackbar_blue(int, void *);
int low_h_b=91, low_s_b=247, low_v_b=47;
int high_h_b=119, high_s_b=255, high_v_b=255;

void on_canny_edge_trackbar_red(int, void *);
int lowThreshold_r = 100;
int ratio_r = 3;
int kernel_size_r = 3;

void on_canny_edge_trackbar_blue(int, void *);
int lowThreshold_b = 100;
int ratio_b = 3;
int kernel_size_b = 3;


/* setup default parameters */
float fball_radius = 0.075;

// Initialization of variable for camera calibration paramters >>change to our own value!!!!
Mat distCoeffs;
float intrinsic_data[9] = {648.494831, 0, 330.912390, 0, 649.675640,246.894210, 0, 0, 1};
float distortion_data[5] = {0.042623, -0.140928, -0.001380, -0.005298,0};

// Initialization of variable for text drawing
double fontScale = 2;
int thickness = 3;
String text;
int iMin_tracking_ball_size = 10;

int deviceID = 0;             // 0 = open default camera
int apiID = cv::CAP_ANY;      // 0 = autodetect default API


#endif
