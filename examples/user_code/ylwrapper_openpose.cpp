
#include <openpose/filestream/headers.hpp>
#include <openpose/wrapper/ylwrapper.hpp>



int main()
{
	std::string image_path = "examples/media/COCO_val2014_000000000192.jpg";
	cv::Mat img_mat = op::loadImage(image_path, CV_LOAD_IMAGE_COLOR);
	
	YLWrapper ylw();
	
	std::cout << "init ready! \n";
	
	std::string res_json = ylw.getPoseEstimation(img_mat);
	
	std::cout << "JSON Result: \n";
	std::cout << res_json << "\n";
	
}
