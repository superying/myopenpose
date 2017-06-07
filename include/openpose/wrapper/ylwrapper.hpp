#ifndef OPENPOSE__WRAPPER__YLWRAPPER_HPP
#define OPENPOSE__WRAPPER__YLWRAPPER_HPP


#include <openpose/core/headers.hpp>
#include <openpose/pose/headers.hpp>


    class YLWrapper
    {
    public:
        /**
         * Constructor.
         */
        YLWrapper();

        std::string getPoseEstimation(cv::Mat oriImg);
        
        void freeGPU();

        

    private:
        cv::Size outputSize;
        cv::Size netInputSize;
        cv::Size netOutputSize;
		op::PoseModel poseModel;
		
		int num_scales = 1;
		float scale_gap = 0.3;
		float alpha_pose = 0.6;
		int num_gpu_start = 0;
		std::string model_folder = "models/";
		
    };


#endif // OPENPOSE__WRAPPER__YLWRAPPER_HPP
