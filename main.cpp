#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

using namespace std;
using namespace ngraph;
namespace ie = InferenceEngine;

// DL Model
// bitsadmin /transfer download https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx %CD%\resnet18-v2-7.onnx
// https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz

// cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release ..
// msbuild onnxapi.sln /p:Configuration=Release

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

    std::shared_ptr<ie::InputInfo> input_info = network.getInputsInfo().begin()->second;
    std::string                    input_name = network.getInputsInfo().begin()->first;
    input_info->setPrecision(ie::Precision::FP32);

    ie::DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(ie::Precision::FP32);

    ie::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
    ie::InferRequest infer_request = executable_network.CreateInferRequest();

    cv::Mat image, inblob_img;
    image = cv::imread("car.png");

    // Set input image to input blob
    auto input_dims = input_info->getTensorDesc().getDims();
    size_t N = input_dims[0];
    size_t C = input_dims[1];
    size_t H = input_dims[2];
    size_t W = input_dims[3];
    //std::cout << N << "," << C << "," << H << "," << W << std::endl;
    cv::resize(image, inblob_img, cv::Size(W,H));
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

    infer_request.Infer();

    float* output = infer_request.GetBlob(output_name)->buffer();
    std::vector<int> idx;
    for(int i=0; i<1000; i++) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [output](const int& left, const int& right) { return output[left] > output[right]; } );
    for (size_t id = 0; id < 5; ++id) {
        std::cout << id+1 <<  " : " << idx[id] << " : " << output[idx[id]]*100 << "% " << labels[idx[id]] << std::endl;
    }
}


// MEMO --------------------------------------------------------------------------------------------------------

// To list all supported ONNX ops in a specific version and domain, use the get_supported_operators as shown in the example below:
/*
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);
for(const auto& op : supported_ops)
{
    std::cout << op << std::endl;
}
*/

// To determine whether a specific ONNX operator in a particular version and domain is supported by the importer, use the is_operator_supported function as shown in the example below:
/*
const std::string op_name = "Abs";
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const bool is_abs_op_supported = ngraph::onnx_import::is_operator_supported(op_name, version, domain);
std::cout << "Abs in version 12, domain `ai.onnx`is supported: " << (is_abs_op_supported ? "true" : "false") << std::endl;
*/
