/**************************************************************************************************************
* EDLines source codes.
* Copyright (C) 2016, Cuneyt Akinlar & Cihan Topal
* E-mails of the authors: cuneytakinlar@gmail.com, cihant@anadolu.edu.tr
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.

* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.

* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
* By using this library you are implicitly assumed to have accepted all of the above statements,
* and accept to cite the following papers:
*
* [1] C. Akinlar and C. Topal, �EDLines: A Real-time Line Segment Detector with a False Detection Control,�
*     Pattern Recognition Letters, 32(13), 1633-1642, DOI: 10.1016/j.patrec.2011.06.001 (2011).
*
* [2] C. Akinlar and C. Topal, �EDLines: Realtime Line Segment Detection by Edge Drawing (ED),�
*     IEEE Int�l Conf. on Image Processing (ICIP), Sep. 2011.
**************************************************************************************************************/

#ifndef _EDLines_
#define _EDLines_

#include "ED.h"
#include "EDColor.h"
#include "NFA.h"

#define SS 0
#define SE 1
#define ES 2
#define EE 3

// light weight struct for Start & End coordinates of the line segment
struct LS {
	cv::Point2d start;
	cv::Point2d end;

	LS(cv::Point2d _start, cv::Point2d _end)
	{
		start = _start;
		end = _end;
	}
};


struct LineSegment {
	double a, b;          // y = a + bx (if invert = 0) || x = a + by (if invert = 1)
	int invert;

	double sx, sy;        // starting x & y coordinates
	double ex, ey;        // ending x & y coordinates

	int segmentNo;        // Edge segment that this line belongs to
	int firstPixelIndex;  // Index of the first pixel within the segment of pixels
	int len;              // No of pixels making up the line segment

	LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len) {
		a = _a;
		b = _b;
		invert = _invert;
		sx = _sx;
		sy = _sy;
		ex = _ex;
		ey = _ey;
		segmentNo = _segmentNo;
		firstPixelIndex = _firstPixelIndex;
		len = _len;
	}
}; 


class EDLines : public ED {
public:
	EDLines(cv::Mat srcImage, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3); // -1 
	EDLines(ED obj, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	EDLines(EDColor obj, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	EDLines();

	std::vector<LS> getLines();
	int getLinesNo();
	cv::Mat getLineImage();
	cv::Mat drawOnImage();

	// EDCircle uses this one 
	static void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo, std::vector<LineSegment> &lines, int min_line_len = 30, double line_error = 1.0);//6
	// add 
	bool ValidateLineSegmentRect(int *x, int *y, LineSegment *ls);
	static void EnumerateRectPoints(double sx, double sy, double ex, double ey,int ptsx[], int ptsy[], int *pNoPoints);
	std::vector<LineSegment> lines;

private:
	std::vector<LineSegment> invalidLines;
	std::vector<LS> linePoints;
	int linesNo;
	int min_line_len;
	double line_error;
	double max_distance_between_two_lines;
	double max_error;
	double prec;
	NFALUT *nfa;
	

	int ComputeMinLineLength();
	void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);
	void JoinCollinearLines();
	
	void ValidateLineSegments();
	bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);
	
	static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
	static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
	static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
	static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
	static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
	static void UpdateLineParameters(LineSegment *ls);

	// Utility math functions
	
};

#endif 
