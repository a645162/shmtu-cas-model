# 上海海事大学 统一认证平台 验证码识别模型训练(PyTorch)

## 本系列项目

### 客户端

* Go Wails版
https://github.com/a645162/SHMTU-Terminal-Wails
* Rust Tauri版(或许以后会做吧~)

### 服务器部署模型

https://github.com/a645162/shmtu-cas-ocr-server

### 统一认证登录流程(数字平台+微信平台)

* Kotlin版(方便移植Android)
https://github.com/a645162/shmtu-cas-kotlin
* Go版(为Wails桌面客户端做准备)
https://github.com/a645162/shmtu-cas-go
* Rust版(未来想做Tauri桌面客户端可能会移植)

### 模型训练

https://github.com/a645162/shmtu-cas-ocr-model

### 模型本地部署

* Windows客户端(包括VC Win32 GUI以及C# WPF)
https://github.com/a645162/shmtu-cas-ocr-demo-windows
* Qt客户端(支持Windows/macOS/Linux)
https://github.com/a645162/shmtu-cas-ocr-demo-qt
* Android客户端
https://github.com/a645162/shmtu-cas-demo-android

## 操作步骤

- 图像转换为灰度图
- 图像二值化
- 对图像7，3分最后一部分作为“等于”与“=”的区分标准
- 使用ResNet-18对最后的符号进行区分，分为3部分(数字，运算符，数字)
- 对1,3部分使用ResNet-18进行**数字**识别
- 对2部分使用ResNet-18进行**运算符**识别