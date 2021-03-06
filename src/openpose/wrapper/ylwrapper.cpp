#include "openpose/wrapper/ylwrapper.hpp"


	YLWrapper::YLWrapper(): outputSize{1280, 720}, netInputSize{656, 368}, netOutputSize{656, 368},
	cvMatToOpInput{netInputSize, num_scales, scale_gap}, cvMatToOpOutput{outputSize}
	{
		poseExtractorCaffe = new op::PoseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel, model_folder, num_gpu_start};
		
		poseExtractorCaffe->initializationOnThread();
	}

	std::string YLWrapper::getPoseEstimation(cv::Mat oriImg) {
		cv::Mat inputImage = oriImg;
		
		// Step 2 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.format(inputImage);
		double scaleInputToOutput;
		op::Array<float> outputArray;
		std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
		
		// Step 3 - Estimate poseKeyPoints
		poseExtractorCaffe->forwardPass(netInputArray, inputImage.size());
		const auto poseKeyPoints = poseExtractorCaffe->getPoseKeyPoints();
		
		const auto numberPeople = poseKeyPoints.getSize(0);
		const auto numberBodyParts = poseKeyPoints.getSize(1);
		    
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
    	//cudaDeviceReset();
    	delete poseExtractorCaffe;
    	 
    }
	

