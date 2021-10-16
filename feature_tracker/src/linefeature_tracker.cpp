#include "linefeature_tracker.h"


LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;
    frame_cnt = 0;
    sum_time = 0.0;
}

void LineFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());

    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    K_ = m_camera->initUndistortRectifyMap(undist_map1_,undist_map2_);    

}

vector<Line> LineFeatureTracker::undistortedLineEndPoints()
{
    vector<Line> un_lines;
    un_lines = curframe_->vecLine;
    float fx = K_.at<float>(0, 0);
    float fy = K_.at<float>(1, 1);
    float cx = K_.at<float>(0, 2);
    float cy = K_.at<float>(1, 2);
    for (unsigned int i = 0; i <curframe_->vecLine.size(); i++)
    {
        un_lines[i].StartPt.x = (curframe_->vecLine[i].StartPt.x - cx)/fx;
        un_lines[i].StartPt.y = (curframe_->vecLine[i].StartPt.y - cy)/fy;
        un_lines[i].EndPt.x = (curframe_->vecLine[i].EndPt.x - cx)/fx;
        un_lines[i].EndPt.y = (curframe_->vecLine[i].EndPt.y - cy)/fy;
    }
    return un_lines;
}


#define USE_EDLINE

// #define USE_LSD
// #define USE_NLT

#ifdef USE_EDLINE

#define MAX_CNT 100
#define SHOW_TRACK 1

bool inImg(const cv::Point2f &pt, cv::Size imSize) {
    const int BORDER_SIZE = 1;
    int x = cvRound(pt.x);
    int y = cvRound(pt.y);
    return BORDER_SIZE <= x && x < imSize.width - BORDER_SIZE && BORDER_SIZE <= y && y < imSize.height - BORDER_SIZE;
}

void visualize_line_match(cv::Mat imageMat1, cv::Mat imageMat2,
                          std::vector<LS> lines_1, std::vector<LS> lines_2,
                          std::vector<std::pair<int, int>> lmatches, std::vector<int> isValid) {
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3)
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    else
        img1 = imageMat1;

    if (imageMat2.channels() != 3)
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    else
        img2 = imageMat2;

    /* plot matches */
    cv::Mat trackImg;
    cv::hconcat(img1, img2, trackImg);     
    for(int i = 0; i < lmatches.size(); i++) {
        if(isValid[i]) {
            LS line_1 = lines_1[lmatches[i].first];
            LS line_2 = lines_2[lmatches[i].second];
            cv::line(trackImg, line_1.start, line_1.end, cv::Scalar(255, 0, 0), 2);
            cv::line(trackImg, cv::Point(line_2.start.x + img1.cols, line_2.start.y), \
                            cv::Point(line_2.end.x + img1.cols, line_2.end.y), cv::Scalar(0, 0, 255), 2);
            cv::line(trackImg, line_1.start, cv::Point(line_2.start.x + img1.cols, line_2.start.y), cv::Scalar(0, 255, 0), 1);
        }
    }
    cv::imshow("trackImg", trackImg);
    cv::waitKey(1);
}

void LineFeatureTracker::readImage(const cv::Mat &_img) {
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;
    cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);
    if (EQUALIZE) {  // 直方图均衡化
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }
    bool first_img = false;
    if (forwframe_ == nullptr) { // 系统初始化的第一帧图像
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;
    }
    int imageWidth = img.cols;
    int imageHeight = img.rows;

    if(curframe_->keyls.size() == 0)  { // first frame
        TicToc t_li;
        EDLines edlines = EDLines(img);
        std::vector<LS> slines = edlines.getLines();
        sum_time += t_li.toc();
        forwframe_->keyls = slines;
        for (size_t i = 0; i < forwframe_->keyls.size(); ++i) 
            forwframe_->lineID.push_back(allfeature_cnt++);
    }
    else {
        TicToc t_track;
        std::vector<cv::Point2f> linePnt;
        for(int i = 0; i < curframe_->keyls.size(); i++) {
            linePnt.push_back(curframe_->keyls[i].start);
            linePnt.push_back(curframe_->keyls[i].end);
        }
        std::vector<cv::Point2f> trackPnt;
        vector< int > lineID_tracked;
        std::vector<uchar> status; std::vector<float> err;
        cv::calcOpticalFlowPyrLK(curframe_->img, forwframe_->img, linePnt, trackPnt, status, err);
        std::vector<LS> linesTrack;
        std::vector<std::pair<int, int>> lmatches; // for visualization
        for(int i = 0; i < curframe_->keyls.size(); i++) {
            if(status[2*i] && status[2*i+1] && inImg(trackPnt[2*i], forwframe_->img.size()) && inImg(trackPnt[2*i+1], forwframe_->img.size())) {
                linesTrack.push_back(LS(trackPnt[2*i], trackPnt[2*i+1]));
                lineID_tracked.push_back(curframe_->lineID[i]);
                lmatches.push_back(std::make_pair(i, linesTrack.size()-1));
            }
        }
        std::vector<LS> linesTmp = linesTrack; // for visualization
        std::vector<int> isValid(linesTrack.size(), 1);
         /* use NFA filter lines */
        TicToc t_NFA;
#define PRECISON_ANGLE 22.5 
        double prec = (PRECISON_ANGLE / 180)*M_PI;
        double prob = 0.125;
#undef PRECISON_ANGLE
        NFALUT *nfa;
        double logNT = 2.0*(log10((double)imageWidth) + log10((double)imageHeight));
        int lutSize = (imageWidth + imageHeight) / 8;
        nfa = new NFALUT(lutSize, prob, logNT); // create look up table

        int noValidLines = 0;
        int *x = new int[(imageWidth + imageHeight) * 4];
        int *y = new int[(imageWidth + imageHeight) * 4];
        for (int i = 0; i< linesTrack.size(); i++) {
            LS ls = linesTrack[i];
            // Compute Line's angle
            double lineAngle = atan2( ( ls.end.y - ls.start.y ), ( ls.end.x - ls.start.x ) );
            if (lineAngle < 0) lineAngle += M_PI;
            int noPoints = 0;
            // Enumerate all pixels that fall within the bounding rectangle
            EDLines::EnumerateRectPoints(ls.start.x, ls.start.y, ls.end.x, ls.end.y, x, y, &noPoints);

            int count = 0, aligned = 0;
            for (int i = 0; i<noPoints; i++) {
                int r = y[i];
                int c = x[i];
                if (r <= 0 || r >= imageHeight - 1 || c <= 0 || c >= imageWidth - 1) continue;
                count++;
                uchar *srcImg = forwframe_->img.data;
                int com1 = srcImg[(r + 1)*imageWidth + c + 1] - srcImg[(r - 1)*imageWidth + c - 1];
                int com2 = srcImg[(r - 1)*imageWidth + c + 1] - srcImg[(r + 1)*imageWidth + c - 1];
                int gx = com1 + com2 + srcImg[r*imageWidth + c + 1] - srcImg[r*imageWidth + c - 1];
                int gy = com1 - com2 + srcImg[(r + 1)*imageWidth + c] - srcImg[(r - 1)*imageWidth + c];
                double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
                double diff = fabs(lineAngle - pixelAngle);
                if (diff <= prec || diff >= M_PI - prec) aligned++;
            }
            bool valid = nfa->checkValidationByNFA(count, aligned);
            if (valid) {
                if (i != noValidLines) {
                    linesTrack[noValidLines] = linesTrack[i];
                    lineID_tracked[noValidLines] = lineID_tracked[i];
                }
                noValidLines++;
            }
            else 
                isValid[i] = 0;
        }
        int size = linesTrack.size();
        for (int i = 1; i <= size - noValidLines; i++) {
            linesTrack.pop_back();
            lineID_tracked.pop_back();
        }

        if(linesTrack.size() < 50) { // 跟踪的线特征少于50了，那就补充新的线特征
            cv::Mat mask = cv::Mat(imageHeight, imageWidth, CV_8UC1, cv::Scalar(255));
            for (auto &it : linesTrack) {
                if (mask.at<uchar>(it.start) == 255)  
                    cv::circle(mask, it.start, LINE_MIN_DIST, 0, -1);
                if (mask.at<uchar>(it.end) == 255)  
                    cv::circle(mask, it.end, LINE_MIN_DIST, 0, -1);
            }
            EDLines newEDLines = EDLines(forwframe_->img);
            std::vector<LS> newlines = newEDLines.getLines();
            if(!mask.empty()) {
                for(size_t keyCounter = 0; keyCounter < newlines.size(); keyCounter++) {
                    LS kl = newlines[keyCounter];
                    if( mask.at<uchar>(kl.start) == 0 && mask.at<uchar>( kl.end ) == 0 ) {
                        newlines.erase( newlines.begin() + keyCounter );
                        keyCounter--;
                    }
                }
            }
            for(auto& line:newlines) {
                linesTrack.push_back(line);
                lineID_tracked.push_back(allfeature_cnt++);
            }
        }
        forwframe_->keyls = linesTrack;
        forwframe_->lineID = lineID_tracked;

        sum_time += t_track.toc();
        mean_time = sum_time/frame_cnt;
        // printf("line  thread ----------------- line feature tracker mean costs: %fms \n", mean_time);
        if(SHOW_TRACK)
            visualize_line_match(curframe_->img, forwframe_->img, curframe_->keyls, linesTmp, lmatches, isValid);
    }
    
    // 将opencv的KeyLine数据转为季哥的Line
    for (int j = 0; j < forwframe_->keyls.size(); ++j) {
        Line l;
        LS kl = forwframe_->keyls[j];
        l.StartPt = kl.start;
        l.EndPt = kl.end;
        l.length = (float) sqrt( pow( kl.start.x - kl.end.x, 2 ) + pow( kl.start.y - kl.end.y, 2 ));
        forwframe_->vecLine.push_back(l);
    }
    curframe_ = forwframe_;
}
#endif

#ifdef USE_LSD
#define MATCHES_DIST_THRESHOLD 30

void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<DMatch> good_matches)
{
    // //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    /* plot matches */
    cv::Mat lm_outImg;
    std::vector<char> l_mask( good_matches.size(), 1 );
    drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lm_outImg, Scalar::all( -1 ), Scalar::all( -1 ), l_mask,
    DrawLinesMatchesFlags::DEFAULT );
    imshow( "lines matches", lm_outImg );
    waitKey(1);
}

void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;

    cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);

//    cv::imshow("lineimg",img);
//    cv::waitKey(1);
    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;
    }

    // step 1: line extraction
    TicToc t_li;
    std::vector<KeyLine> lsd, keylsd;
    Ptr<LSDDetector> lsd_;
    lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
    lsd_->detect( img, lsd, 2, 2 );

    sum_time += t_li.toc();
//    ROS_INFO("line detect costs: %fms", t_li.toc());

    Mat lbd_descr, keylbd_descr;
    // step 2: lbd descriptor
    TicToc t_lbd;
    Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor( );
    bd_->compute( img, lsd, lbd_descr );

//////////////////////////
    for ( int i = 0; i < (int) lsd.size(); i++ )
    {
        if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
        {
            keylsd.push_back( lsd[i] );
            keylbd_descr.push_back( lbd_descr.row( i ) );
        }
    }
//    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
    sum_time += keylsd.size() * t_lbd.toc() / lsd.size();
///////////////

    forwframe_->keylsd = keylsd;
    forwframe_->lbd_descr = keylbd_descr;

    for (size_t i = 0; i < forwframe_->keylsd.size(); ++i) {
        if(first_img)
            forwframe_->lineID.push_back(allfeature_cnt++);
        else
            forwframe_->lineID.push_back(-1);   // give a negative idvisualize_line_match
    }
    if(curframe_->keylsd.size() > 0)
    {

        /* compute matches */
        TicToc t_match;
        std::vector<DMatch> lsd_matches;
        Ptr<BinaryDescriptorMatcher> bdm_;
        bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);
//        ROS_INFO("lbd_macht costs: %fms", t_match.toc());
        sum_time += t_match.toc();
        mean_time = sum_time/frame_cnt;
        ROS_INFO("line feature tracker mean costs: %fms", mean_time);

        /* select best matches */
        std::vector<DMatch> good_matches;
        std::vector<KeyLine> good_Keylines;
        good_matches.clear();
        for ( int i = 0; i < (int) lsd_matches.size(); i++ )
        {
            if( lsd_matches[i].distance < MATCHES_DIST_THRESHOLD ) {

                DMatch mt = lsd_matches[i];
                KeyLine line1 =  forwframe_->keylsd[mt.queryIdx] ;
                KeyLine line2 =  curframe_->keylsd[mt.trainIdx] ;
                Point2f serr = line1.getStartPoint() - line2.getStartPoint();
                Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                if((serr.dot(serr) < 60 * 60) && (eerr.dot(eerr) < 60 * 60))   // 线段在图像里不会跑得特别远
                    good_matches.push_back( lsd_matches[i] );
            }

        }

        std::cout << forwframe_->lineID.size() <<" " <<curframe_->lineID.size();
        for (int k = 0; k < good_matches.size(); ++k) {
            DMatch mt = good_matches[k];
            forwframe_->lineID[mt.queryIdx] = curframe_->lineID[mt.trainIdx];

        }
        visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, good_matches);

        vector<KeyLine> vecLine_tracked, vecLine_new;
        vector< int > lineID_tracked, lineID_new;
        Mat DEscr_tracked, Descr_new;

        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < forwframe_->keylsd.size(); ++i)
        {
            if( forwframe_->lineID[i] == -1)
            {
                forwframe_->lineID[i] = allfeature_cnt++;
                vecLine_new.push_back(forwframe_->keylsd[i]);
                lineID_new.push_back(forwframe_->lineID[i]);
                Descr_new.push_back( forwframe_->lbd_descr.row( i ) );
            }
            else
            {
                vecLine_tracked.push_back(forwframe_->keylsd[i]);
                lineID_tracked.push_back(forwframe_->lineID[i]);
                DEscr_tracked.push_back( forwframe_->lbd_descr.row( i ) );
            }
        }
        int diff_n = 50 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
        if( diff_n > 0)    // 补充线条
        {

            for (int k = 0; k < vecLine_new.size(); ++k) {
                vecLine_tracked.push_back(vecLine_new[k]);
                lineID_tracked.push_back(lineID_new[k]);
                DEscr_tracked.push_back(Descr_new.row(k));
            }

        }

        forwframe_->keylsd = vecLine_tracked;
        forwframe_->lineID = lineID_tracked;
        forwframe_->lbd_descr = DEscr_tracked;

    }

    // 将opencv的KeyLine数据转为季哥的Line
    for (int j = 0; j < forwframe_->keylsd.size(); ++j) {
        Line l;
        KeyLine lsd = forwframe_->keylsd[j];
        l.StartPt = lsd.getStartPoint();
        l.EndPt = lsd.getEndPoint();
        l.length = lsd.lineLength;
        forwframe_->vecLine.push_back(l);
    }
    curframe_ = forwframe_;
}
#endif


#ifdef  USE_NLT
void LineFeatureTracker::NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines,
                                            vector<pair<int, int> > &lineMatches) {
    float th = 3.1415926/9;
    float dth = 30 * 30;
    for (size_t i = 0; i < forw_lines.size(); ++i) {
        Line lf = forw_lines.at(i);
        Line best_match;
        size_t best_j = 100000;
        size_t best_i = 100000;
        float grad_err_min_j = 100000;
        float grad_err_min_i = 100000;
        vector<Line> candidate;

        // 从 forw --> cur 查找
        for(size_t j = 0; j < cur_lines.size(); ++j) {
            Line lc = cur_lines.at(j);
            // condition 1
            Point2f d = lf.Center - lc.Center;
            float dist = d.dot(d);
            if( dist > dth) continue;  //
            // condition 2
            float delta_theta1 = fabs(lf.theta - lc.theta);
            float delta_theta2 = 3.1415926 - delta_theta1;
            if( delta_theta1 < th || delta_theta2 < th)
            {
                //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                candidate.push_back(lc);
                //float cost = fabs(lf.image_dx - lc.image_dx) + fabs( lf.image_dy - lc.image_dy) + 0.1 * dist;
                float cost = fabs(lf.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                //std::cout<< "line match cost: "<< cost <<" "<< cost - sqrt( dist )<<" "<< sqrt( dist ) <<"\n\n";
                if(cost < grad_err_min_j)
                {
                    best_match = lc;
                    grad_err_min_j = cost;
                    best_j = j;
                }
            }

        }
        if(grad_err_min_j > 50) continue;  // 没找到

        //std::cout<< "!!!!!!!!! minimal cost: "<<grad_err_min_j <<"\n\n";

        // 如果 forw --> cur 找到了 best, 那我们反过来再验证下
        if(best_j < cur_lines.size())
        {
            // 反过来，从 cur --> forw 查找
            Line lc = cur_lines.at(best_j);
            for (int k = 0; k < forw_lines.size(); ++k)
            {
                Line lk = forw_lines.at(k);

                // condition 1
                Point2f d = lk.Center - lc.Center;
                float dist = d.dot(d);
                if( dist > dth) continue;  //
                // condition 2
                float delta_theta1 = fabs(lk.theta - lc.theta);
                float delta_theta2 = 3.1415926 - delta_theta1;
                if( delta_theta1 < th || delta_theta2 < th)
                {
                    //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                    //candidate.push_back(lk);
                    //float cost = fabs(lk.image_dx - lc.image_dx) + fabs( lk.image_dy - lc.image_dy) + dist;
                    float cost = fabs(lk.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                    if(cost < grad_err_min_i)
                    {
                        grad_err_min_i = cost;
                        best_i = k;
                    }
                }

            }
        }

        if( grad_err_min_i < 50 && best_i == i){

            //std::cout<< "line match cost: "<<grad_err_min_j<<" "<<grad_err_min_i <<"\n\n";
            lineMatches.push_back(make_pair(best_j,i));
        }
        /*
        vector<Line> l;
        l.push_back(lf);
        vector<Line> best;
        best.push_back(best_match);
        visualizeLineTrackCandidate(l,forwframe_->img,"forwframe_");
        visualizeLineTrackCandidate(best,curframe_->img,"curframe_best");
        visualizeLineTrackCandidate(candidate,curframe_->img,"curframe_");
        cv::waitKey(0);
        */
    }

}


void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;
    cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);
    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;
    }

    // step 1: line extraction
    TicToc t_li;
    int lineMethod = 2;
    bool isROI = false;
    lineDetector ld(lineMethod, isROI, 0, (float)img.cols, 0, (float)img.rows);
    //ROS_INFO("ld inition costs: %fms", t_li.toc());
    TicToc t_ld;
    forwframe_->vecLine = ld.detect(img);

    for (size_t i = 0; i < forwframe_->vecLine.size(); ++i) {
        if(first_img)
            forwframe_->lineID.push_back(allfeature_cnt++);
        else
            forwframe_->lineID.push_back(-1);   // give a negative id
    }
    ROS_INFO("line detect costs: %fms", t_ld.toc());

    // step 3: junction & line matching
    if(curframe_->vecLine.size() > 0)
    {
        TicToc t_nlt;
        vector<pair<int, int> > linetracker;
        NearbyLineTracking(forwframe_->vecLine, curframe_->vecLine, linetracker);
        // ROS_INFO("line match costs: %fms", t_nlt.toc());

        // 对新图像上的line赋予id值
        for(int j = 0; j < linetracker.size(); j++)
        {
            forwframe_->lineID[linetracker[j].second] = curframe_->lineID[linetracker[j].first];
        }

        // show NLT match
        visualizeLineMatch(curframe_->vecLine, forwframe_->vecLine, linetracker,
                           curframe_->img, forwframe_->img, "NLT Line Matches", 10, true,
                           "frame");
        visualizeLinewithID(forwframe_->vecLine,forwframe_->lineID,forwframe_->img,"forwframe_");
        visualizeLinewithID(curframe_->vecLine,curframe_->lineID,curframe_->img,"curframe_");
        stringstream ss;
        ss <<"/home/hyj/datasets/line/" <<frame_cnt<<".jpg";
        // SaveFrameLinewithID(forwframe_->vecLine,forwframe_->lineID,forwframe_->img,ss.str().c_str());
        waitKey(5);


        vector<Line> vecLine_tracked, vecLine_new;
        vector< int > lineID_tracked, lineID_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < forwframe_->vecLine.size(); ++i)
        {
            if( forwframe_->lineID[i] == -1)
            {
                forwframe_->lineID[i] = allfeature_cnt++;
                vecLine_new.push_back(forwframe_->vecLine[i]);
                lineID_new.push_back(forwframe_->lineID[i]);
            }
            else
            {
                vecLine_tracked.push_back(forwframe_->vecLine[i]);
                lineID_tracked.push_back(forwframe_->lineID[i]);
            }
        }
        int diff_n = 30 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
        if( diff_n > 0)    // 补充线条
        {
            for (int k = 0; k < vecLine_new.size(); ++k) {
                vecLine_tracked.push_back(vecLine_new[k]);
                lineID_tracked.push_back(lineID_new[k]);
            }
        }

        forwframe_->vecLine = vecLine_tracked;
        forwframe_->lineID = lineID_tracked;
    }
    curframe_ = forwframe_;
}
#endif