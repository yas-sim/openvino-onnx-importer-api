#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

using namespace std;
using namespace ngraph;
namespace ie = InferenceEngine;

int main(void) {

    // Read labels from file
    std::string labelFileName = "synset_words.txt";
    std::cout << "Loading class label file : " << labelFileName << std::endl;
    std::vector<std::string> labels;
    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            trim(strLine);
            labels.push_back(strLine);
        }
    }

    // Read an ONNX model file
    const std::string onnx_path = "model.onnx";
    std::cout << "Importing an ONNX model : " << onnx_path << std::endl;
    const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(onnx_path);

    ie::Core ie;

    // Wraps nGraph model with IE::CNNNetwork
    std::cout << "Converting an ONNX model into CNNNetwork" << std::endl;
    ie::CNNNetwork network(ng_function);

    // Setup for input blobs
    std::shared_ptr<ie::InputInfo> input_info = network.getInputsInfo().begin()->second;
    std::string                    input_name = network.getInputsInfo().begin()->first;
    input_info->setPrecision(ie::Precision::FP32);

    // Setup for output blobs
    ie::DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(ie::Precision::FP32);

    // Load the model to IE core and create an infer request object
    ie::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
    ie::InferRequest infer_request = executable_network.CreateInferRequest();

    cv::Mat image, inblob_img;
    image = cv::imread("car.png");

    // Set input image to input blob (resize, BGR->RGB, NHWC->HCHW)
    // Both algorithm generates the same result
#if 0
    auto input_dims = input_info->getTensorDesc().getDims();  // 0,1,2,3 = N,C,H,W
    cv::resize(image, inblob_img, cv::Size(input_dims[3], input_dims[2]));
    cv::cvtColor(inblob_img, inblob_img, cv::COLOR_BGR2RGB);
    uint8_t* buf = inblob_img.data;
    float* inblob = infer_request.GetBlob(input_name)->buffer();
    for(int h=0; h<H; h++) {
        for(int w=0; w<W; w++) {
            inblob[0 * (W*H) + h*W + w] = (float)(*buf++)/255.f;    
            inblob[1 * (W*H) + h*W + w] = (float)(*buf++)/255.f;    
            inblob[2 * (W*H) + h*W + w] = (float)(*buf++)/255.f;    
        }
    }
#else
    auto input_dims = input_info->getTensorDesc().getDims();  // 0,1,2,3 = N,C,H,W
    cv::resize(image, inblob_img, cv::Size(input_dims[3], input_dims[2]));
    cv::cvtColor(inblob_img, inblob_img, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> planes;
    cv::split(inblob_img, planes);
    size_t img_size = input_dims[2] * input_dims[3];  // H*W
    float* inblob = infer_request.GetBlob(input_name)->buffer();
    for(size_t ch=0; ch<input_dims[1]; ch++) {
        uint8_t* srcptr = planes[ch].data;
        for(size_t i=0; i<img_size; i++) {
            *inblob++ = (float)(*srcptr++)/255.f;
        }
    }
#endif

    // Inference
    infer_request.Infer();

    // Display inference output
    float* output = infer_request.GetBlob(output_name)->buffer();
    std::vector<int> idx;
    for(int i=0; i<1000; i++) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [output](const int& left, const int& right) { return output[left] > output[right]; } );
    for (size_t id = 0; id < 5; ++id) {
        std::cout << id+1 <<  " : " << idx[id] << " : " << output[idx[id]]*100 << "% " << labels[idx[id]] << std::endl;
    }
}
