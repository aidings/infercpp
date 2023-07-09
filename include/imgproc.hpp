#include "opencv2/opencv.hpp"


class ImgProc
{
public:
    cv::Mat imread(const std::string& path, int flag=cv::IMREAD_COLOR);
    {
        cv::Mat image = cv::Mat();
        try{
            image = cv::imread(path, flag);
        }
        catch (const cv::Exception& e) {
            std::cerr << "error loading the image\n";
        }
        return image;
    };

};

class AugmentClassify
{
public:
    AugmentClassify(const cv::Size& dst_size, const cv::Rect& roi, const cv::Scalar& mean, const cv::Scalar& std)
    {
        m_dst_size = dst_size;
        m_roi = roi;
        m_mean = mean;
        m_std = std;
    };

    cv::Mat operator()(const cv::Mat& image)
    {
        cv::Mat dst = cv::Mat();
        try{
            cv::Mat roi = image(m_roi);
            cv::resize(roi, dst, m_dst_size);
            dst.convertTo(dst, CV_32FC3, 1.0/255.0);
            dst = dst - m_mean;
            dst = dst / m_std;
        }
        catch (const cv::Exception& e) {
            std::cerr << "error augmenting the image\n";
        }
        return dst;
    };

private:
    cv::Size m_dst_size;
    cv::Rect m_roi;
    cv::Scalar m_mean;
    cv::Scalar m_std;
};