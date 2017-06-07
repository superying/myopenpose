#ifndef OPENPOSE__WRAPPER__YLWRAPPER_HPP
#define OPENPOSE__WRAPPER__YLWRAPPER_HPP


#include <openpose/core/headers.hpp>
#include <openpose/pose/headers.hpp>

namespace op
{
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
		
		op::CvMatToOpInput cvMatToOpInput;
		op::CvMatToOpOutput cvMatToOpOutput;
		op::PoseExtractorCaffe poseExtractorCaffe;
		op::PoseRenderer poseRenderer;
		
    };
}





// Implementation
#include <openpose/core/headers.hpp>
#include <openpose/pose/headers.hpp>
namespace op
{
	YLWrapper::YLWrapper() {
		outputSize.width = 1280;
		outputSize.height = 720;
		netInputSize.width = 656;
		netInputSize.height = 368;
		netOutputSize = netInputSize;
		poseModel = op::PoseModel::COCO_18;
		
		cvMatToOpInput{netInputSize, num_scales, scale_gap};
		cvMatToOpOutput{outputSize};
		poseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
												  model_folder, num_gpu_start};
		poseRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
		
		poseExtractorCaffe.initializationOnThread();
		poseRenderer.initializationOnThread();
	}

	std::string YLWrapper::getPoseEstimation(cv::Mat oriImg) {
		cv::Mat inputImage = oriImg;
		
		// Step 2 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.format(inputImage);
		double scaleInputToOutput;
		op::Array<float> outputArray;
		std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
		
		// Step 3 - Estimate poseKeyPoints
		poseExtractorCaffe.forwardPass(netInputArray, inputImage.size());
		const auto poseKeyPoints = poseExtractorCaffe.getPoseKeyPoints();
		
		float scale = 1.0/scaleInputToOutput;
		    
		//generate json object
		std::string res_json = "";

		res_json += "{\n";
		res_json +=  "\"version\":0.1,\n";
		res_json +=  "\"bodies\":[\n";
		for (int ip=0;ip<numberPeople;ip++) {
			res_json +=  "{\n";
			res_json +=  "\"joints\":";
			res_json +=  "[";
			for (int ij=0;ij<numberBodyParts;ij++) {
				res_json += std::to_string(scale*poseKeyPoints[ip*numberBodyParts*3 + ij*3+0]);
				res_json += ",";
				res_json += std::to_string(scale*poseKeyPoints[ip*numberBodyParts*3 + ij*3+1]);
				res_json += ",";
				res_json += std::to_string(poseKeyPoints[ip*numberBodyParts*3 + ij*3+2]);
				if (ij<numberBodyParts-1) res_json += ",";
			}
			res_json += "]\n";
			res_json += "}";
			if (ip<numberPeople-1) {
				res_json += ",\n";
			}
		}
		res_json += "]\n";
		res_json += "}\n";
		
		return res_json;
		
	}

    void YLWrapper::freeGPU() {
    	
    }
	
}

#endif // OPENPOSE__WRAPPER__YLWRAPPER_HPP