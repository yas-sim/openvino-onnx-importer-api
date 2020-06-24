
# Overview
[ONNX importer API](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_OnnxImporterTutorial.html) is introduced from Intel(r) Distribution of OpenVINO(tm) toolkit 2020.3 LTS version. It allows user to load an ONNX model and convert it into an nGraph model. Furthermore, user can import the nGraph model to OpenVINO CNNNetwork model so that the user can use a familiar Inference Engine API to run the model.  
This project will demonstrate how to use the ONNX importer API.  
```
ONNX model -(ONNX importer API)-> nGraph model -> CNNNetwork (an Inference Engine object)
```

[ONNX importer API](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_OnnxImporterTutorial.html)はIntel(r) Distribution of OpenVINO(tm) toolkit 2020.3 LTSバージョンから導入されたAPIです。これを使うことによってONNXモデルを読み込み、nGraphモデルに変換することが可能になります。また、このnGraphモデルをInference EngineのCNNNetworkモデルに変換することで使い慣れたInference Engine APIを使った推論が可能になります。  
このプロジェクトではONNX importer APIの使い方の例を示します。  

## 1. Prerequisites
* [Intel Distribution of OpenVINO toolkit 2020.3 LTS (or above)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

## 2. Preparing required files (ONNX model, class label text file, input image)

Linux
```sh
# Download an ONNX model (ResNet-50)
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx
cp resnet50-v2-7.onnx model.onnx
# Download a class label text file
wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt
# Copy an input image
cp ${INTEL_OPENVINO_DIR}/deployment_tools/demo/car.png .
```

Windows
```sh
# Download an ONNX model (ResNet-50)
bitsadmin /transfer download https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx %CD%\resnet50-v2-7.onnx
copy resnet50-v2-7.onnx model.onnx
# Download a class label text file
bitsadmin /transfer download https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt %CD%\synset_words.txt
# Copy an input image
copy "%INTEL_OPENVINO_DIR%\deployment_tools\demo\car.png" .
```

## 3. How To Build

Linux
```sh
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
```

Windows
```sh
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release ..
msbuild onnx-importer.sln /p:Configuration=Release
cd ..
```

### 4. How to Run

This sample program supports image classification models such as Googlenet, ResNet, Squeezenet, Mobilenet and so on.  

Linux
```sh
$ ./build/onnx-importer
```

Windows
```sh
C:> build\Release\onnx-importer.exe
```

## 5. Test Environment
- Ubuntu 18.04 / Windows 10 1909  
- OpenVINO 2020.3 LTS  

### Example of output log

```sh
C:>build\release\onnx-importer.exe
Loading class label file : synset_words.txt
Importing an ONNX model : model.onnx
Converting an nGraph model into CNNNetwork
1 : 817 : 1112.83% n04285008 sports car, sport car
2 : 511 : 1082.38% n03100240 convertible
3 : 479 : 999.68% n02974003 car wheel
4 : 436 : 996.005% n02814533 beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
5 : 656 : 875.367% n03770679 minivan
```

### 'car.png'
![car](./resources/car.png)

## See Also  
* [ONNX Inporter API Tutorial](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_OnnxImporterTutorial.html)  
* [Build a Model with nGraph Library](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_nGraphTutorial.html)
* [ONNX runtime](https://github.com/microsoft/onnxruntime)
