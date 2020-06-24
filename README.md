

## Preparing required files (ONNX model, class label text file, input image)

Linux
```sh
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx
cp resnet50-v2-7.onnx model.onnx
wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt
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

## How To Build

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

### How to Run

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
Converting an ONNX model into CNNNetwork
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
