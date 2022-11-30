#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
// #include <opencv2/gapi/own/types.hpp>
#include <stdio.h>
#include <ros/ros.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <zed_interfaces/ObjectsStamped.h>
#include <zed_interfaces/Object.h>
#include <cv_bridge/cv_bridge.h>

using namespace nvinfer1;
using std::string;

const int inputNum = 3 * 256 * 192;
const int outputNum = 26;
const int batch = 1;

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;

cv::Mat imageForShow;

float* data;
float* prob;

float* buffers[2];

int inputIndex = 0;
int outputIndex = 1;


cudaStream_t stream;
cudaError_t err = cudaStreamCreate(&stream);

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override
	{
		if (severity != Severity::kINFO) {
			std::cout << msg << std::endl;
		}
	}
} logger;

cv::Mat resizeImage(cv::Mat image, bool keepRatio, int targetWidth, int targetHeight)
{
	int originWidth = image.cols;
	int originHeight = image.rows;
	float scaleWidth = 1.0;
	float scaleHeight = 1.0;
	cv::Mat targetImage(192, 256, CV_8UC3);
	if (keepRatio == true) {
		int imageSizeMin = std::min(originWidth, originHeight);
		int imageSizeMax = std::max(originWidth, originHeight);
		int targetSizeMin = std::min(targetWidth, targetHeight);
		int targetSizeMax = std::max(targetWidth, targetHeight);
		float scale = (float)(targetSizeMin * 1.0 / imageSizeMin);
		if (((int)scale * imageSizeMax) > targetSizeMax) {
			scale = (float)(targetSizeMax * 1.0 / imageSizeMax);
		}
		scaleWidth = scale;
		scaleHeight = scale;
	} else {
		scaleWidth = (float)(targetWidth * 1.0 / originWidth);
		scaleHeight = (float)(targetHeight * 1.0 / originHeight);
	}
	cv::resize(image, targetImage, cv::Size(0,0), scaleWidth, scaleHeight);
 	return targetImage;	
}

cv::Mat normalization(cv::Mat image)
{
	std::vector<float> mean{0.485, 0.456, 0.406};
	std::vector<float> std{0.229, 0.224, 0.225};
	cv::Mat result;
	// convert dtype uchar -> float32
	image.convertTo(result, CV_32F, 1.0/255.0, 0.0);

	std::vector<cv::Mat> bgrChannels(3);
	cv::split(result, bgrChannels);
	for (auto i = 0; i < bgrChannels.size(); ++i){
		bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std[i], (0.0 - mean[i]) / std[i]);
	}
	cv::merge(bgrChannels, result);
	return result;
}

int argmax(float* res, int startIdx, int endIdx) 
{
	int ans = startIdx;
	float min = *(res + startIdx);
	for (int i = startIdx + 1; i <= endIdx; ++i) {
		if (min > *(res + i)) {
			ans = i;
		}
	}
	return (ans - startIdx);
}

std::vector<string> postProcess(float* res)
{
	std::vector<string> age_list = {"AgeOver60", "Age18-60", "AgeLess18"};
	std::vector<string> direct_list = {"Front", "Side", "Back"};
	std::vector<string> bag_list = {"HandBag", "ShoulderBag", "BackBag"};
	std::vector<string> upper_list = {"UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice"};
	std::vector<string> lower_list = {"LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress"};
	float glasses_threshold = 0.3;
	float hold_threshold = 0.6;
	float threshold = 0.5;
	std::vector<string> res_labels(18);
	// gender
	res_labels[0] = *(res + 22) > threshold ? "Female" : "Male";
	// age
	res_labels[1] = age_list[argmax(res, 19, 21)];
	// direct
	res_labels[2] = direct_list[argmax(res, 23, 25)];
	// glasses
	res_labels[2] = *(res + 1) > glasses_threshold ? "glasses" : "None";
	// hat
	res_labels[3] = *(res + 0) > threshold ? "hat" : "None";
	// hold objects
	res_labels[4] = *(res + 18) > hold_threshold ? "holdObjects" : "None";
	// bag
	res_labels[5] = bag_list[argmax(res, 15, 17)];
	// sleeve
	res_labels[6] = *(res + 3) > *(res + 2) ? "LongSleeve" : "ShortSleeve";
	// upper label
	res_labels[7] = *(res + 4) > threshold ? upper_list[0] : "None";
	res_labels[8] = *(res + 5) > threshold ? upper_list[1] : "None";
	res_labels[9] = *(res + 6) > threshold ? upper_list[2] : "None";
	res_labels[10] = *(res + 7) > threshold ? upper_list[3] : "None";
	// lower label
	res_labels[11] = *(res + 8) > threshold ? lower_list[0] : "None";
	res_labels[12] = *(res + 9) > threshold ? lower_list[1] : "None";
	res_labels[13] = *(res + 10) > threshold ? lower_list[2] : "None";
	res_labels[14] = *(res + 11) > threshold ? lower_list[3] : "None";
	res_labels[15] = *(res + 12) > threshold ? lower_list[4] : "None";
	res_labels[16] = *(res + 13) > threshold ? lower_list[5] : "None";
	// shoes
	res_labels[17] = *(res + 14) > threshold ? "Boots" : "No boots";
	return res_labels;
}

void imageBodyCallback(sensor_msgs::ImageConstPtr image, zed_interfaces::ObjectsStampedConstPtr detections)
{
	auto persons = detections->objects;
	auto img = cv_bridge::toCvShare(image, "bgr8");
	cv::Mat imageCvMat;
	std::vector<cv::Rect> regions;
	std::vector<cv::Mat> ROIs;
	std::vector<std::vector<string>> objectsAttributions;
	imageForShow = img->image;
	if (persons.size() == 0) {
		std::cout << "No Person detected!" << std::endl;
		return;
	}
	cv::cvtColor(img->image, imageCvMat, cv::COLOR_BGR2RGB);
	imageCvMat = normalization(imageCvMat);
	std::cout << "Detected Person Numbers: " << persons.size() << std::endl;
	std::cout << "Exact Person region images" << std::endl;
	for (zed_interfaces::Object& person : persons) {
		auto bbox = person.bounding_box_2d;
		int x = bbox.corners[0].kp[0];
		int y = bbox.corners[0].kp[1];
		int width = bbox.corners[2].kp[0] - bbox.corners[0].kp[0];
		int height = bbox.corners[2].kp[1] - bbox.corners[0].kp[1];
		// std::cout << "x: " << x << " y: " << y << " w: " << width <<" h: " << height << std::endl;
		cv::Rect region(x, y, width, height);
		regions.push_back(region);
		cv::Mat ROI = imageCvMat(region);
		ROIs.push_back(ROI);
	}
	std::cout << "Moving image to GPU memory" << std::endl;
	for (cv::Mat& roi : ROIs) {
		// std::cout << roi.rows << " " << roi.cols << std::endl;
		if (roi.rows == 0 || roi.cols == 0) {
			return;
		}

		// permute and transit image data to buffer
		int area = 256 * 192;
		float* matPtr = imageCvMat.ptr<float>(0);
		for (int c = 0; c < 3; ++c) {
			for (int row = 0; row < 256; ++row) {
				for (int col = 0; col < 192; ++col) {
					int index = c * area + row * 192 + col;
					int divider = index / 3;
					for (int i = 0; i < 3; ++i) {
						data[divider + i * area] = static_cast<float>(matPtr[index]);
					}
				}
			}
		}

		std::cout << "Inference attributions" << std::endl;
		cudaMemcpyAsync(buffers[inputIndex], data, inputNum * sizeof(float), cudaMemcpyHostToDevice, stream);
		context->enqueue(batch, (void**)buffers, stream, nullptr);
		cudaMemcpyAsync(prob, buffers[1], outputNum * sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		std::cout << "Inference Done" << std::endl;
		for (int i = 0; i < 26; ++i) {
			std::cout << (float)*(prob + i);
		}
		std::cout << std::endl;
		std::vector<string> attributions = postProcess(prob);
		objectsAttributions.push_back(attributions);
	}

	if (objectsAttributions.size() == 0) {
		std::cerr << "detection error ..." << std::endl;
		std::cerr << "please check your code!" << std::endl;
		return;
	}

	for (int i = 0; i < regions.size(); ++i) {
		cv::rectangle(imageCvMat, regions[i], cv::Scalar(0, 255, 0, 255), 1, 8, 0);
		auto attributions = objectsAttributions[i];
		int counts = 0;
		int leftBottom_X = regions[i].x;
		int leftBottom_Y = regions[i].y + regions[i].height;
		// for (string& attribution : attributions) {
		// 	cv::putText(img->image, attribution, cv::Point(leftBottom_X, leftBottom_Y + counts * 5), cv::FONT_HERSHEY_SIMPLEX, 12, cv::Scalar(0, 0, 255, 255));
		// 	++counts;
		// }
	}
	
	for (string& attribution : objectsAttributions[0]) {
		std::cout << attribution;
	}
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "human_attribution");
	ros::NodeHandle nh;

	std::cout << "Checking engine file path......" << std::endl;
    std::string engine_name = "/home/nvidia/Projects/wheelchair_ws/src/human_tracking_opencv/model/human_attribute_recognition.engine";
	std::ifstream file;
	file.open(engine_name, std::ios::binary);
	if (!file.good()) {
		std::cerr << "file path error!" << std::endl;
	}
	else {
		std::cout << "engine file exist!" << std::endl;
	}
	
	std::cout << "Loading engine......" << std::endl;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	char *modelStream = new char[size];
	file.read(modelStream, size);
	file.close();
	std::cout << "Load engine success!" << std::endl;

	std::cout << "creating InferRuntime" << std::endl;
	runtime = createInferRuntime(logger);
	engine = runtime->deserializeCudaEngine(modelStream, size);
	context = engine->createExecutionContext();
	delete[] modelStream;
	std::cout << "create InferRuntime success" << std::endl;

	std::cout << "creating host and gpu memory" << std::endl;
	data = (float*)malloc(inputNum * sizeof(float));
	prob = (float*)malloc(outputNum * sizeof(float));
	inputIndex = engine->getBindingIndex("image");
	outputIndex = engine->getBindingIndex("sigmoid_0.tmp_0");
	cudaMalloc((void**)&buffers[inputIndex], inputNum * sizeof(float));
	cudaMalloc((void**)&buffers[outputIndex], outputNum * sizeof(float));
	std::cout << "created host and gpu memory success" << std::endl;

	message_filters::Subscriber<sensor_msgs::Image> image_sub_(nh, "/zed2/zed_node/rgb/image_rect_color", 1);
    message_filters::Subscriber<zed_interfaces::ObjectsStamped> Objects_sub_(nh, "/zed2/zed_node/obj_det/objects", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, zed_interfaces::ObjectsStamped> sync(image_sub_, Objects_sub_, 10);
    sync.registerCallback(boost::bind(&imageBodyCallback, _1, _2));

	ros::Rate rate(30);
	while (ros::ok()) {
		if (imageForShow.rows > 0) {
			cv::imshow("Attributions Detection Results", imageForShow);
			cv::waitKey(10);
		}
		ros::spinOnce();
		rate.sleep();
	}
	return 0;
}
