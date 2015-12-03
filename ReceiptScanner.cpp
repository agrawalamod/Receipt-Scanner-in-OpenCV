/* 
@author
Amod Agrawal (2013125) and Shuchita Gupta (2013101)
Image Analysis Project - 2013
CSE340
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <map>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdint>
#define PI 3.14159265
#define e 2.718281828

using namespace cv;
using namespace std;
Mat IDFT(Mat complexI);
Mat DFTMagnitude(Mat complexI, Size s);
Mat FFT(Mat I);
Mat alphaMeanTrim(Mat img1, int alpha);
Mat Erosion(Mat src, int erosion_size, int erosion_elem );
Mat Dilation( Mat src, int dilation_size, int dilation_elem);
vector<Point2f> distance(Mat img, vector<Point> polygon);
Mat warpImage(Mat src, vector<Point2f> corners);

Mat FFT(Mat I)
{

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    padded.convertTo(padded,CV_32F);
    for(int i =0; i<padded.size().height; i++)
    {
      for(int j=0; j<padded.size().width; j++)
      {
        padded.at<float>(i,j) = padded.at<float>(i,j) * pow(-1,i+j);
      }
    }

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix
    //DFTMagnitude(complexI, padded.size());

    return complexI;

    
}
Mat IDFT(Mat complexI)
{
    cv::Mat inverseTransform;
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    for(int i =0; i<inverseTransform.size().height; i++)
    {
      for(int j=0; j<inverseTransform.size().width; j++)
      {
        inverseTransform.at<float>(i,j) = inverseTransform.at<float>(i,j) / pow(-1,i+j);
      }
    }
    
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    //inverseTransform.convertTo(inverseTransform,CV_8U);

    namedWindow("Inverse Fourier Transform", CV_WINDOW_AUTOSIZE); 
    imshow("Inverse Fourier Transform", inverseTransform); 
    waitKey(0);
    //destroyWindow("Inverse Fourier Transform");
  
    return inverseTransform;

}

Mat DFTMagnitude(Mat complexI, Size s)
{
    Mat planes[] = {Mat::zeros(s, CV_32F), Mat::zeros(s, CV_32F)};
    split(complexI, planes);                 // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    
    namedWindow("DFT Magnitude", CV_WINDOW_AUTOSIZE); 
    imshow("DFT Magnitude", magI); 
    waitKey(0);
    destroyWindow("DFT Magnitude");
    
    return magI;

}

Mat convolution_fourier(Mat FFT_img, Mat filter)
{

  Size s = filter.size();
  Mat resultComplex;
  Mat planes[] = {Mat::zeros(s, CV_32F), Mat::zeros(s, CV_32F)};
  split(FFT_img, planes);
  Mat real = planes[0];
  Mat im = planes[1];
  for(int i=0; i<real.size().height; i++)
  {
    for(int j=0; j<real.size().width; j++)
    {
      planes[0].at<float>(i,j) = real.at<float>(i,j) * filter.at<float>(i,j);
      planes[1].at<float>(i,j) = im.at<float>(i,j) * filter.at<float>(i,j);
               
    }
  }
  merge(planes, 2, resultComplex);
  //DFTMagnitude(resultComplex, real.size());
  return resultComplex; 
}
Mat initializeMat(int n, int arr[5][5])
{
    Mat A = Mat::zeros(n, n, CV_8UC1);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            //cout << arr[i][i] << endl;
            A.at<uchar>(i,j) = arr[i][j];
        }
    }
    return A;

}
Mat GaussianLPF(float D, int height, int width)
{
    Mat filter = Mat::zeros(height, width, CV_32F);
    int origin_x = width/2;
    int origin_y = height/2;
    double Dtemp, value;
    for(int i=0; i <height; i++)
    {
        for(int j=0; j<width; j++)
        {
            Dtemp = sqrt(pow(j-origin_x,2) + pow(i-origin_y,2));
            //cout << Dtemp << endl;
            double numerator = -pow(Dtemp,2);
            double denominator = 2 * pow(D,2);
            value = pow(e,numerator/denominator);
            //cout << numerator << " " << denominator << " "  << value <<endl;
            filter.at<float>(i,j) = value;
        }
    }
    //Mat result;
    //filter.convertTo(result, CV_8U);
    imshow("Gaussian LPF", filter);
    waitKey(0);

    return filter;

}
Mat GaussianHPF(float D, int height, int width)
{
    Mat filter = Mat::zeros(height, width, CV_32F);
    int origin_x = width/2;
    int origin_y = height/2;
    double Dtemp, value;
    for(int i=0; i <height; i++)
    {
        for(int j=0; j<width; j++)
        {
            Dtemp = sqrt(pow(j-origin_x,2) + pow(i-origin_y,2));
            //cout << Dtemp << endl;
            double numerator = -pow(Dtemp,2);
            double denominator = 2 * pow(D,2);
            value = 1 -pow(e,numerator/denominator);
            //cout << numerator << " " << denominator << " "  << value <<endl;
            filter.at<float>(i,j) = value;
        }
    }
    //Mat result;
    //filter.convertTo(result, CV_8U);
    imshow("Gaussian HPF", filter);
    waitKey(0);

    return filter;

}
Mat ButterworthLPF(float D, int height, int width, int n)
{
    Mat filter = Mat::zeros(height, width, CV_32F);
    int origin_x = width/2;
    int origin_y = height/2;
    double Dtemp, value;
    for(int i=0; i <height; i++)
    {
        for(int j=0; j<width; j++)
        {
            Dtemp = sqrt(pow(j-origin_x,2) + pow(i-origin_y,2));
            //cout << Dtemp << endl;
            float ratio = pow((Dtemp/D),2*n);
            value = 1/(1+ratio);
            filter.at<float>(i,j) = value;
        }
    }
    
    imshow("Butterworth LPF", filter);
    //waitKey(0);

    return filter;

}
Mat ButterworthHPF(float D, int height, int width, int n)
{
    Mat filter = Mat::zeros(height, width, CV_32F);
    int origin_x = width/2;
    int origin_y = height/2;
    double Dtemp, value;
    for(int i=0; i <height; i++)
    {
        for(int j=0; j<width; j++)
        {
            Dtemp = sqrt(pow(j-origin_x,2) + pow(i-origin_y,2));
            //cout << Dtemp << endl;
            float ratio = pow((D/Dtemp),2*n);
            value = 1/(1+ratio);
            filter.at<float>(i,j) = value;
        }
    }
    
    imshow("Butterworth HPF", filter);
    waitKey(0);

    return filter;

}
Mat CannyThreshold(int lowThreshold, int highThreshold, Mat img1)
{
    int ratio = 3;
    Mat detected_edges;
    //int lowThreshold = 75;
    //int highThreshold = 200;
    int kernel_size = 3;
    //Reduce noise with a kernel 3x3
    blur(img1, detected_edges, Size(3,3));

    //Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, highThreshold, kernel_size);

    //Using Canny's output as a mask, we display our result
    imshow( "Canny Output",detected_edges );
    imwrite("canny_result.jpg",detected_edges);
    waitKey(0);
    return detected_edges;
 }
 Mat Erosion(Mat src, int erosion_size, int erosion_elem )
{
  int erosion_type;
  Mat erosion_dst;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( src, erosion_dst, element );
  cout << "Erosion done! \n";
  imshow( "Erosion Demo", erosion_dst );
  waitKey(0);

  return erosion_dst;
}

/** @function Dilation */
Mat Dilation( Mat src, int dilation_size, int dilation_elem)
{
  int dilation_type;
  Mat dilation_dst;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( src, dilation_dst, element );
  cout << "Dilation done! \n";
  imshow( "Dilation Demo", dilation_dst );
  waitKey(0);

  return dilation_dst;
}
Mat warpImage(Mat src, vector<Point2f> corners)
{
    float x1,x2,x3,x4,y1,y2,y3,y4;
    x1 = corners[0].x;
    y1 = corners[0].y;
    x2 = corners[1].x;
    y2 = corners[1].y;
    x3 = corners[2].x;
    y3 = corners[2].y;
    x4 = corners[3].x;
    y4 = corners[3].y;

    float minx, maxx, miny, maxy;
   /* minx = x1<x4 ? x1:x4;
    maxx = x2>x3 ? x2:x3;
    miny = y1<y2 ? y1:y2;
    maxy = y3>y4 ? y3:y4;*/

    minx = x1>x4 ? x1:x4;
    maxx = x2<x3 ? x2:x3;
    miny = y1>y2 ? y1:y2;
    maxy = y3<y4 ? y3:y4;

    cout << minx << " " << maxx << " " << miny << " " << maxy<<endl;
    // Define the destination image
    cv::Mat quad = cv::Mat::zeros(maxy-miny, maxx-minx, CV_8UC1);
    //src.convertTo(src,CV_32F);

    // Corners of the destination image
    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));

    // Get transformation matrix
    cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);

    // Apply perspective transformation
    cv::warpPerspective(src, quad, transmtx, quad.size());
    cv::imshow("quadrilateral", quad);
    imwrite("warpedImage.jpg", quad);
 
    //Display input and output
    //imshow("Input",input);
    //imshow("Output",output);
 
    waitKey(0);
    return quad;
    //return output;
}
 vector<Point> findLargestContours(Mat src)
 {
    int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;
    Mat thr(src.rows,src.cols,CV_8UC1); 
    Mat dst(src.rows,src.cols,CV_8UC1,Scalar::all(0));
    threshold(src, src,25, 255,THRESH_BINARY); //Threshold the gray
  
    vector<vector <Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
 
    findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
   
    for( int i = 0; i<contours.size(); i++) // iterate through each contour. 
    {
       double a=contourArea(contours[i],false);  //  Find the area of contour
       if(a>largest_area)
       {
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
            bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
       }
   
    }
    //cout << largest_area << " " << src.rows*src.cols;
    Scalar color( 255,255,255);
    Mat x = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat contour_image;
    src.copyTo(contour_image);
    drawContours( contour_image, contours,largest_contour_index, color, 5,8, hierarchy);
    vector<vector<Point> > contours_poly(1);
    approxPolyDP( Mat(contours[largest_contour_index]), contours_poly[0],10, true );
    


     // Draw the largest contour using previously stored index.
    //rectangle(src, bounding_rect,  Scalar(255,255,255),5, 8,0);  
    //imshow( "src", x );
    
    
    
    imshow("Largest Contour", contour_image);
    imwrite("largest_contour.jpg",contour_image);
    waitKey(0);
    

    return contours_poly[0];
    
}

vector <Point2f> detectCorners(Mat src, vector<Point> contours_poly)
{
    
    vector <Point2f> corners = distance(src,contours_poly);

    Mat all_points;
    Mat corners_points;
    src.copyTo(all_points);
    src.copyTo(corners_points);
    for(int i=0; i<contours_poly.size(); i++)
    {
        circle(all_points, contours_poly[i], 20, Scalar(255,255,255), -1);

    }
    for(int i=0; i<corners.size(); i++)
    {
        circle(corners_points, corners[i], 20, Scalar(255,255,255), -1);

    }
    imwrite("all_points.jpg", all_points);
    imwrite("corners_points.jpg",corners_points);

    return corners;

}

vector<Point2f> distance(Mat img, vector<Point> polygon)
{
    vector <Point2f> coord;
    //cout << img.rows << " " << img.cols << endl;

    double min = sqrt(pow(img.cols,2) + pow(img.rows,2));
    int index;

    for(int i=0; i <polygon.size(); i++)
    {
        double dist = sqrt(pow(polygon[i].x,2) + pow(polygon[i].y,2));
        if(dist <= min)
        {
            min = dist;
            index =i;
             
            
        }
     }
     coord.push_back(Point2f(polygon[index].x,polygon[index].y));

    min = sqrt(pow(img.cols,2) + pow(img.rows,2)); 
    for(int i=0; i <polygon.size(); i++)
    {
        double dist = sqrt(pow(polygon[i].x-img.cols,2) + pow((polygon[i].y),2));
        if(dist <= min)
        {
            min = dist;
            index =i;

            
        }
     }
     coord.push_back(Point2f(polygon[index].x,polygon[index].y));

     min = sqrt(pow(img.cols,2) + pow(img.rows,2)); 
    for(int i=0; i <polygon.size(); i++)
    {
        double dist = sqrt(pow((polygon[i].x-img.cols),2) + pow((polygon[i].y-img.rows),2));
        if(dist <= min)
        {
            min = dist;
            index =i;
            
        }
     }
     coord.push_back(Point2f(polygon[index].x,polygon[index].y));

     min = sqrt(pow(img.cols,2) + pow(img.rows,2)); 
    for(int i=0; i <polygon.size(); i++)
    {
        double dist = sqrt(pow(polygon[i].x,2) + pow((polygon[i].y-img.rows),2));
        //cout << polygon[i] << ": " << dist<< "\n";
        if(dist <=min)
        {
            min = dist;
            index =i; 
        }       
     }
     coord.push_back(Point2f(polygon[index].x,polygon[index].y));

    cout<< coord;

    return coord;

    //polygon.erase(polygon.begin()+index);
    //coord.push_back(x,y);




}


int main(int argc, const char** argv)
{
    
     
    if(argc!=2)
    {
        cout << " Program_name <Img1.jpg>";
        return -1;
    }
    Mat img1;

    img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); //read the image data in the file "MyPic.JPG" and store it in 'img'
    if (img1.empty()) //check whether the image is loaded or not
    {
        cout << "Error : Image cannot be loaded!" << endl;
        return -1;
    }
    namedWindow("Image Selected", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
    imshow("Image Selected", img1); //display the image which is stored in the 'img' in the "MyWindow" window
    waitKey(0);
    destroyWindow("Image Selected");

    /*float Do = 150;
    Mat GaussianFilterLPF = GaussianLPF(Do,img1.size().height,img1.size().width);
    Mat fft_img = FFT(img1);
    Mat result;
    cout << "Gaussian LPF\n";
    result = convolution_fourier(fft_img, GaussianFilterLPF);
    IDFT(result);*/

    Mat CannyResult = CannyThreshold(75, 200, img1);
    

    //Dilation(Erosion(CannyResult,1,0),1,0);
    vector<Point> polygon = findLargestContours(CannyResult);
    vector<Point2f> corners = detectCorners(CannyResult, polygon);

    //Mat again = CannyThreshold(75, 200, img1);
    Mat warpedImage = warpImage(img1,corners);      

    //blur(warpedImage, warpedImage, Size(3,3));
    Mat dst = Mat::zeros(warpedImage.rows, warpedImage.cols, CV_8UC1);

    adaptiveThreshold(warpedImage, dst,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 30);
    
    //Mat eroded = Erosion(dst,1,0);
    //Mat dilated = Dilation(eroded,1,0);
    
     


    imshow("dst", dst);
    imwrite("threshold_image1.jpg", dst);
    waitKey(0);
    //Dilation(Erosion(coutour_output,3,0),3,0);









    return 0;
    
}