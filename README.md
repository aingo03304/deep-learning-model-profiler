# Deep Learning Model Profiler

Deep Learning Model Profiler (DLMP) is easy to use tool for profiling various models.  
It provides PyTorch/`tensorflow` Model Zoo, and also custom model with `tensorflow` SavedModel or PyTorch saved model.

## Requirements
```sh
pip install -r requirements.txt
```

## Supported Models
Because supported models vary depending on the framework, some models cannot be supported on some framwork. Instead of providing only the models used in both frameworks, DLMP provides the feature that you can put the saved model to be profiled.

Here are supported models.

- AlexNet: `pytorch`
- VGG-11: `pytorch`
- VGG-13: `pytorch`
- VGG-16: `pytorch`, `tensorflow`
- VGG-19: `pytorch`, `tensorflow`
- VGG-11-bn: `pytorch`
- VGG-13-bn: `pytorch`
- VGG-16-bn: `pytorch`
- VGG-19-bn: `pytorch`
- ResNet-18: `pytorch`
- ResNet-34: `pytorch`
- ResNet-50: `pytorch`, `tensorflow`
- ResNet-101: `pytorch`, `tensorflow`
- ResNet-152: `pytorch`, `tensorflow`
- ResNet-50-v2: `tensorflow`
- ResNet-101-v2: `tensorflow`
- ResNet-152-v2: `tensorflow`
- SqueezeNet-1.0: `pytorch`
- SqueezeNet-1.1: `pytorch`
- Densenet-121: `pytorch`, `tensorflow`
- Densenet-169: `pytorch`, `tensorflow`
- Densenet-201: `pytorch`, `tensorflow`
- Densenet-161: `pytorch`
- Inception-v3: `pytorch`, `tensorflow`
- Inception-ResNet-v2: `tensorflow`
- Xception: `tensorflow`
- GoogleNet: `pytorch`
- MobileNet: `tensorflow`
- MobileNet-v2: `pytorch`, `tensorflow`
- ResNeXt-50-32x4d: `pytorch`
- ResNeXt-101-32x8d: `pytorch`
- Wide-ResNet-50-2: `pytorch`
- Wide-ResNet-101-2: `pytorch`
- NASNet-mobile: `tensorflow`
- NASNet-large: `tensorflow`
- MNASNet-1.0: `pytorch`
- EfficientNet-B0: `tensorflow`
- EfficientNet-B1: `tensorflow`
- EfficientNet-B2: `tensorflow`
- EfficientNet-B3: `tensorflow`
- EfficientNet-B4: `tensorflow`
- EfficientNet-B5: `tensorflow`
- EfficientNet-B6: `tensorflow`
- EfficientNet-B7: `tensorflow`