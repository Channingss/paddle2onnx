目前paddle2onnx工具集支持 20+ PaddlePaddle的Operator转化成ONNX的Operator，目前主要支持的转化的模型主要是有两大类，图像分类和图像检测。
受限于不同框架的差异，部分模型可能会存在目前无法转换的情况，，如若您发现无法转换或转换失败，存在较大diff等问题，欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/paddle-onnx/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会即时跟进：）

# 图像分类
目前支持的图像分类模型在下表中，模型的来源主要是来自于PaddleHub的CV库和PaddlePaddle的CV库，其它的图像分类模型的转化没有验证，如有需要可以在issue进行需求申请，我们会需求评估并进行反馈。

| 模型 | 来源 | operator version|
|-------|--------|---------|
| ResNet | [resnet_v2_50_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet&en_category=ImageClassification) |9|
| DenseNet | [densenet_121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) |9|
| ShuffleNet | [shufflenet_v2_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=shufflenet_v2_imagenet&en_category=ImageClassification) |9|
| MobileNet| [mobilenet_v2_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_imagenet&en_category=FeatureExtraction) |9|
| VGG16| [vgg16_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=vgg16_imagenet&en_category=ImageClassification) |9|
| SE_ResNext50| [se_resnext50_32x4d_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=se_resnext50_32x4d_imagenet&en_category=ImageClassification) |9|
| InceptionV4| [InceptionV4_imagenet](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) |9|
| SE_ResNet50_vd| [SE_ResNet50_vd_imagenet](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar) |9|
| SqueezeNet1_0| [SqueezeNet1_0_imagenet](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_0_pretrained.tar) |9|
| Xception41| [Xception41_imagenet](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_pretrained.tar) |9|

# 图像检测
目前paddle2onnx支持的模型是有SSD、YoloV3模型，目前ONNX对检测模型不能完全支持，受限于这个原因，paddle2onnx对检测模型也不能完全支持。后续我们计划增加对其它检测模型的支持，基于ONNX目前对检测模型支持的现状，将会集中于一阶段检测模型。

| 模型 | 来源 | operator version|
|-------|--------|---------|
|SSD_MobileNet|[ssd_mobilenet_v1_pascal](https://www.paddlepaddle.org.cn/hubdetail?name=ssd_mobilenet_v1_pascal&en_category=ObjectDetection) |9|
|YoloV3_DarkNet53|[yolov3_darknet_voc](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |9|
|YoloV3_ResNet34|[yolov3_r34_voc](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |9|
|YoloV3_MobileNet-V1|[yolov3_mobilenet_v1_voc](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |9|
