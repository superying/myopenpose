#include <chrono>
#include <openpose/filestream/headers.hpp>
#include <openpose/wrapper/ylwrapper.hpp>



int main()
{
	std::string image_path = "examples/media/COCO_val2014_000000000192.jpg";
	cv::Mat img_mat = op::loadImage(image_path, CV_LOAD_IMAGE_COLOR);
	
	YLWrapper ylw;
	
	//std::cout << "init ready! \n";
	
	const auto timerBegin = std::chrono::high_resolution_clock::now();
	
	for(int i=0; i<100; i++) {
		std::string res_json = ylw.getPoseEstimation(img_mat);
	
		std::cout << "index: " << i << "\n";
		std::cout << "JSON Result: \n";
		std::cout << res_json << "\n";
	}
	
	ylw.freeGPU();
	
	//const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
	
	//std::cout << "Total time: " << std::to_string(totalTimeSec) << " seconds.";
	
	
	cv::Mat img_mat2 = op::loadImage(image_path, CV_LOAD_IMAGE_COLOR);
		
	YLWrapper ylw2;
	
	for(int j=0; j<100; j++) {
		std::string res_json2 = ylw2.getPoseEstimation(img_mat2);
	
		std::cout << "index: " << j << "\n";
		std::cout << "JSON Result: \n";
		std::cout << res_json2 << "\n";
	}
	
	
	const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timerBegin).count() * 1e-9;
		
	std::cout << "Total time: " << std::to_string(totalTimeSec) << " seconds.";
	
	
}
