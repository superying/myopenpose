
#include "openpose/wrapper/ylwrapper.hpp"


	YLWrapper::YLWrapper(int gpu_id) {
		
		outputSize.width = 1280;
		outputSize.height = 720;
		netInputSize.width = 656;
		netInputSize.height = 368;
		netOutputSize = netInputSize;
		poseModel = op::PoseModel::COCO_18;
		
		
		op::CvMatToOpInput icvMatToOpInput{netInputSize, num_scales, scale_gap};
		op::CvMatToOpOutput icvMatToOpOutput{outputSize};
		op::PoseExtractorCaffe iposeExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
												  model_folder, num_gpu_start};
		op::PoseRenderer iposeRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
		
		cvMatToOpInput = icvMatToOpInput;
		cvMatToOpOutput = icvMatToOpOutput;
		poseExtractorCaffe = iposeExtractorCaffe;
		poseRenderer = iposeRenderer;
		
		
		/*
		cvMatToOpInput{netInputSize, num_scales, scale_gap};
		cvMatToOpOutput{outputSize};
		poseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
														  model_folder, num_gpu_start};
		poseRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
		*/
		
		poseExtractorCaffe.initializationOnThread();
		poseRenderer.initializationOnThread();
		
	}

	std::string YLWrapper::getPoseEstimation(cv::Mat oriImg) {
		cv::Mat inputImage = oriImg;
		
		//test
		/*
		op::CvMatToOpInput cvMatToOpInput{netInputSize, num_scales, scale_gap};
		op::CvMatToOpOutput cvMatToOpOutput{outputSize};
		op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
												  model_folder, num_gpu_start};
		op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
		
		poseExtractorCaffe.initializationOnThread();
		poseRenderer.initializationOnThread();		
		*/
		
		
		
		// Step 2 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.format(inputImage);
		double scaleInputToOutput;
		op::Array<float> outputArray;
		std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
		
		// Step 3 - Estimate poseKeyPoints
		poseExtractorCaffe.forwardPass(netInputArray, inputImage.size());
		const auto poseKeyPoints = poseExtractorCaffe.getPoseKeyPoints();
		
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
    	
    }
	

