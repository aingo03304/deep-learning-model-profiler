import torchvision.models as torchmodels
import tensorflow.keras.applications as tfmodels

MODEL_LIST = {
    'AlexNet': {
        'pytorch': torchmodels.alexnet,
    },
    'VGG-11': {
        'pytorch': torchmodels.vgg11,
    },
    'VGG-13': {
        'pytorch': torchmodels.vgg13,
    },
    'VGG-16': {
        'pytorch': torchmodels.vgg16,
        'tensorflow': tfmodels.VGG16,
    },
    'VGG-19': {
        'pytorch': torchmodels.vgg19,
        'tensorflow': tfmodels.VGG19,
    },
    'VGG-11-bn': {
        'pytorch': torchmodels.vgg11_bn,
    },
    'VGG-13-bn': {
        'pytorch': torchmodels.vgg13_bn,
    },
    'VGG-16-bn': {
        'pytorch': torchmodels.vgg16_bn,
    },
    'VGG-19-bn': {
        'pytorch': torchmodels.vgg19_bn,
    },
    'ResNet-18': {
        'pytorch': torchmodels.restnet18,
    },
    'ResNet-34': {
        'pytorch': torchmodels.restnet34,
    },
    'ResNet-50': {
        'pytorch': torchmodels.restnet50,
        'tensorflow': tfmodels.ResNet50,
    },
    'ResNet-101': {
        'pytorch': torchmodels.restnet101,
        'tensorflow': tfmodels.ResNet101,
    },
    'ResNet-152': {
        'pytorch': torchmodels.restnet152,
        'tensorflow': tfmodels.ResNet152,
    },
    'ResNet-50-v2': {
        'tensorflow': tfmodels.ResNet50V2,
    },
    'ResNet-101-v2': {
        'tensorflow': tfmodels.ResNet101V2,
    },
    'ResNet-152-v2': {
        'tensorflow': tfmodels.ResNet152V2,
    },
    'SqueezeNet-1.0': {
        'pytorch': torchmodels.squeezenet1_0,
    },
    'SqueezeNet-1.1': {
        'pytorch': torchmodels.squeezenet1_1,
    },
    'Densenet-121': {
        'pytorch': torchmodels.densenet121,
        'tensorflow': tfmodels.DenseNet121,
    },
    'Densenet-169': {
        'pytorch': torchmodels.densenet169,
        'tensorflow': tfmodels.DenseNet169,
    },
    'Densenet-201': {
        'pytorch': torchmodels.densenet201,
        'tensorflow': tfmodels.DenseNet201,
    },
    'Densenet-161': {
        'pytorch': torchmodels.densenet161,
    },
    'Inception-v3': {
        'pytorch': torchmodels.inception_v3,
        'tensorflow': tfmodels.InceptionV3,
    },
    'Inception-ResNet-v2': {
        'tensorflow': tfmodels.InceptionResNetV2,
    },
    'Xception': {
        'tensorflow': tfmodels.Xception,
    },
    'GoogleNet': {
        'pytorch': torchmodels.googlenet,
    },
    'MobileNet': {
        'tensorflow': tfmodels.MobileNet,
    },
    'MobileNet-v2': {
        'pytorch': torchmodels.mobilenet_v2,
        'tensorflow': tfmodels.MobileNetV2,
    },
    'ResNeXt-50-32x4d': {
        'pytorch': resnext50_32x4d,
    },
    'ResNeXt-101-32x8d': {
        'pytorch': resnext50_32x8d,
    },
    'Wide-ResNet-50-2': {
        'pytorch': torchmodels.wide_resnet50_2,
    },
    'Wide-ResNet-101-2': {
        'pytorch': torchmodels.wide_resnet101_2,
    },
    'NASNet-mobile': {
        'tensorflow': tfmodels.NASNetMobile,
    },
    'NASNet-large': {
        'tensorflow': tfmodels.NASNetLarge,
    },
    'MNASNet-1.0': {
        'pytorch': torchmodels.mnasnet1_0,
    },
    'EfficientNet-B0': {
        'tensorflow': tfmodels.EfficientNetB0,
    },
    'EfficientNet-B1': {
        'tensorflow': tfmodels.EfficientNetB1,
    },
    'EfficientNet-B2': {
        'tensorflow': tfmodels.EfficientNetB2,
    },
    'EfficientNet-B3': {
        'tensorflow': tfmodels.EfficientNetB3,
    },
    'EfficientNet-B4': {
        'tensorflow': tfmodels.EfficientNetB4,
    },
    'EfficientNet-B5': {
        'tensorflow': tfmodels.EfficientNetB5,
    },
    'EfficientNet-B6': {
        'tensorflow': tfmodels.EfficientNetB6,
    },
    'EfficientNet-B7': {
        'tensorflow': tfmodels.EfficientNetB7,
    },
}

def get_model_by_sign(model_sign):
    model_name, framework = model_sign.split(":")
    return MODEL_LIST[model_name][framework]