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
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

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
    const cv::Size windowedSize = outputSize;
    op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
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
    // Step 4 - Render poseKeyPoints
    poseRenderer.renderPose(outputArray, poseKeyPoints);
    // Step 5 - OpenPose output format to cv::Mat
    auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    frameDisplayer.displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Running openPoseTutorialPose1
    return openPoseTutorialPose1();
}
