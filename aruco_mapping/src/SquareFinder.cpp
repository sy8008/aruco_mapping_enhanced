#pragma once

#include "math/Quadrilateral.cpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * SquareFinder can be used to detect distorted squares in images.
 */
class SquareFinder
{
	public:
		/**
		 * Detect quads in grayscale image.
		 * @param gray Grayscale image.
		 * @param limitCosine Limit value for cosine in the quad corners, by default its 0.6.
		 * @param maxError Max error percentage relative to the square perimeter.
		 * @returns sequence of squares detected on the image the sequence is stored in the specified memory storage
		 */
		static vector<Quadrilateral> findSquares(Mat gray, double limitCosine = 0.6, int minArea = 100,int maxArea=20000, double maxError = 0.025)
		{
			//Quads found
			vector<Quadrilateral> squares = vector<Quadrilateral>();

			//Contours
			vector<vector<Point>> contours;
            Mat pyr,img;
            pyrDown(gray, pyr, Size(gray.cols/2, gray.rows/2));
            pyrUp(pyr, img, gray.size());
            Canny(img, pyr, 10, 120, 5);
            imshow("Canny",pyr);
            imshow("Gray",gray);
			//Find contours and store them all as a list
			findContours(pyr, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
			vector<Point> approx;

			for(unsigned int i = 0; i < contours.size(); i++)
			{
				//Approximate contour with accuracy proportional to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * maxError, true);

				//Square contours have 4 vertices after approximation relatively large area (to filter out noisy contours)and be convex.
				if(approx.size() == 4 && fabs(contourArea(Mat(approx))) > minArea && isContourConvex(Mat(approx)) && fabs(contourArea(Mat(approx)))<maxArea)
				{
					float maxCosine = 0;

					//Find the maximum cosine of the angle between joint edges
					for(int j = 2; j < 5; j++)
					{
						float cosine = fabs(angleCornerPointsCos(approx[j%4], approx[j-2], approx[j-1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					//Check if all angle corner close to 90 (more than the max cosine)
					if(maxCosine < limitCosine)
					{
						Quadrilateral quad = Quadrilateral();
						
						for(int j = 0; !approx.empty() && j < 4; j++)
						{
							quad.points[j] = approx.back();
							approx.pop_back();
						}

						squares.push_back(quad);
					}
				}
			}

			return squares;
		}

		static double angle( Point pt1, Point pt2, Point pt0 )
        {
            double dx1 = pt1.x - pt0.x;
            double dy1 = pt1.y - pt0.y;
            double dx2 = pt2.x - pt0.x;
            double dy2 = pt2.y - pt0.y;
            return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
        }

        static void findSquaresCV( const Mat& image, vector<vector<Point> >& squares,int thresh1 = 50, int thresh2=200,int N = 11)
    {
        squares.clear();
//        imshow("cvsquare raw",image);
        Mat pyr, timg, gray;
        // down-scale and upscale the image to filter out the noise
        pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
        pyrUp(pyr, timg, image.size());
        vector<vector<Point> > contours;

            for( int l = 0; l < N; l++ )
            {
                // hack: use Canny instead of zero threshold level.
                // Canny helps to catch squares with gradient shading
                if( l == 0 )
                {
                    // apply Canny. Take the upper threshold from slider
                    // and set the lower to 0 (which forces edges merging)
                    Canny(image, gray, thresh1, thresh2, 5);
                    // dilate canny output to remove potential
                    // holes between edge segments
                    dilate(gray, gray, Mat(), Point(-1,-1));
                    imshow("Canny",gray);
                    imshow("gray0",image);
                    waitKey(1);
                }
                else
                {
                    // apply threshold if l!=0:
                    //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                    gray = image >= (l+1)*255/N;


//                    imshow("Canny2",gray);

                }
                // find contours and store them all as a list
                findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
                vector<Point> approx;
                // test each contour
                for( size_t i = 0; i < contours.size(); i++ )
                {
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    if( approx.size() == 4 &&
                        fabs(contourArea(approx)) > 1000 &&
                        isContourConvex(approx) )
                    {
                        double maxCosine = 0;
                        for( int j = 2; j < 5; j++ )
                        {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                            maxCosine = MAX(maxCosine, cosine);
                        }
                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if( maxCosine < 0.5 )
                            squares.push_back(approx);
                    }
                }
            }
        }

		/**
		 * Draw quads into the matrix.
		 * 
		 * @param mat Mat to draw quads.
		 * @param quads Vector of quadrilaterals to draw.
		 */
		static void drawQuads(Mat mat, vector<Quadrilateral> quads)
		{
			for(unsigned int i = 0; i < quads.size(); i++)
			{
				for(unsigned int j = 0; j < 4; j++)
				{
					line(mat, quads[i].points[j], quads[i].points[(j + 1) % 4], Scalar(255, 0, 255), 2);
				}
			}
		}

		/**
		 * Finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2.
		 *
		 * @param b Point b.
		 * @param c Point c.
		 * @param a Point a.
		 */
		static float angleCornerPointsCos(Point b, Point c, Point a)
		{
			float dx1 = b.x - a.x;
			float dy1 = b.y - a.y;
			float dx2 = c.x - a.x;
			float dy2 = c.y - a.y;

			return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-12);
		}
};
