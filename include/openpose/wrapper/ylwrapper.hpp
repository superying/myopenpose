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
        //const cv::Size outputSize(1280, 720);
		//const cv::Size netInputSize(656, 368);
		//cv::Size netOutputSize = netInputSize;
        cv::Size outputSize;
        cv::Size netInputSize;
        cv::Size netOutputSize;
		op::PoseModel poseModel;
		
		int num_scales = 1;
		float scale_gap = 0.3;
		float alpha_pose = 0.6;
		int num_gpu_start = 0;
		std::string model_folder = "models/";
		
		
		/*
		op::CvMatToOpInput cvMatToOpInput{netInputSize, num_scales, scale_gap};
		op::CvMatToOpOutput cvMatToOpOutput{outputSize};
		op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
			  model_folder, num_gpu_start};
		op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
		*/
    };


#endif // OPENPOSE__WRAPPER__YLWRAPPER_HPP
