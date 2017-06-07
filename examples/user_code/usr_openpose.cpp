// ------------------------- OpenPose Library Tutorial - Pose - Example 1 - Extract from Image -------------------------
// This first example shows the user how to:
    // 1. Load an image (`filestream` module)
    // 2. Extract the pose of that image (`pose` module)
    // 3. Render the pose on a resized copy of the input image (`pose` module)
    // 4. Display the rendered pose (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module: for the Array<float> class that the `pose` module needs
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
//#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
//#include <openpose/utilities/headers.hpp>

int openPoseTutorialPose1()
{
    // ------------------------- INITIALIZATION -------------------------
    cv::Size outputSize;
    cv::Size netInputSize;
    cv::Size netOutputSize;
    op::PoseModel poseModel = op::PoseModel::COCO_18;
    outputSize.width = 1280;
    outputSize.height = 720;
    netInputSize.width = 656;
	netInputSize.height = 368;
	netOutputSize = netInputSize;
	int num_scales = 1;
	float scale_gap = 0.3;
	float alpha_pose = 0.6;
	int num_gpu_start = 0;
	std::string model_folder = "models/";
	std::string image_path = "examples/media/COCO_val2014_000000000192.jpg";
	
    
    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cvMatToOpInput{netInputSize, num_scales, scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, num_scales, scale_gap, poseModel,
                                              model_folder, num_gpu_start};
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};
    //const cv::Size windowedSize = outputSize;
    //op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorCaffe.initializationOnThread();
    poseRenderer.initializationOnThread();

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    cv::Mat inputImage = op::loadImage(image_path, CV_LOAD_IMAGE_COLOR); // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);
    
    // Step 2 - Format input image to OpenPose input and output formats
    const auto netInputArray = cvMatToOpInput.format(inputImage);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
    // Step 3 - Estimate poseKeyPoints
    poseExtractorCaffe.forwardPass(netInputArray, inputImage.size());
    const auto poseKeyPoints = poseExtractorCaffe.getPoseKeyPoints();
    
    //test poseKeyPoints
    std::cout << "scaleInputToOutput: " << scaleInputToOutput << "\n";
    std::cout << "poseKeyPoints \n";
    std::cout << "number of people: " << poseKeyPoints.getSize(0) << "\n";
    std::cout << "number of body parts: " << poseKeyPoints.getSize(1) << "\n";
    
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
    
    
	std::cout << "JSON Result: \n";
	std::cout << res_json << "\n";
	
    
    // Step 4 - Render poseKeyPoints
    //poseRenderer.renderPose(outputArray, poseKeyPoints);
    
    //test outputArray
    //std::cout << "outputArray \n";
    //std::cout << "number of people: " << outputArray.getSize(0) << "\n";
    //std::cout << "number of body parts: " << outputArray.getSize(1) << "\n";
    
    
    // Step 5 - OpenPose output format to cv::Mat
    //auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    //frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    // Return successful message
    return 0;
}

int main()
{
    // Running openPoseTutorialPose1
    return openPoseTutorialPose1();
}
