import torch
import torch.optim as torchoptim
import tensorflow
import tensorflow.keras.optimizers as tfoptim
import torchvision.models as torchmodels
import tensorflow.keras.applications as tfmodels
from exception import (
    UnsupportedFrameworkError,
)
from typing import Union

FRAMEWORK_LIST = ['pytorch', 'tensorflow']

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
        'pytorch': torchmodels.resnet18,
    },
    'ResNet-34': {
        'pytorch': torchmodels.resnet34,
    },
    'ResNet-50': {
        'pytorch': torchmodels.resnet50,
        'tensorflow': tfmodels.ResNet50,
    },
    'ResNet-101': {
        'pytorch': torchmodels.resnet101,
        'tensorflow': tfmodels.ResNet101,
    },
    'ResNet-152': {
        'pytorch': torchmodels.resnet152,
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
        'pytorch': torchmodels.resnext50_32x4d,
    },
    'ResNeXt-101-32x8d': {
        'pytorch': torchmodels.resnext101_32x8d,
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

OPTIM_LIST = {
    'Adadelta': {
        'pytorch': torchoptim.Adadelta,
        'tensorflow': tfoptim.Adadelta,
    },
    'Adagrad': {
        'pytorch': torchoptim.Adagrad,
        'tensorflow': tfoptim.Adagrad,
    },
    'Adam': {
        'pytorch': torchoptim.Adam,
        'tensorflow': tfoptim.Adam,
    },
    'AdamW': {
        'pytorch': torchoptim.AdamW,
    },
    'SparseAdam': {
        'pytorch': torchoptim.SparseAdam,
    },
    'Adamax': {
        'pytorch': torchoptim.Adamax,
        'tensorflow': tfoptim.Adamax,
    },
    'ASGD': {
        'pytorch': torchoptim.ASGD,
    },
    'LBFGS': {
        'pytorch': torchoptim.LBFGS,
    },
    'RMSprop': {
        'pytorch': torchoptim.RMSprop,
        'tensorflow': tfoptim.RMSprop,
    },
    'Rprop': {
        'pytorch': torchoptim.Rprop,
    },
    'SGD': {
        'pytorch': torchoptim.SGD,
        'tensorflow': tfoptim.SGD
    },
    'Ftrl': {
        'tensorflow': tfoptim.Ftrl,
    },
    'Nadam': {
        'tensorflow': tfoptim.Nadam,
    }
}

def check_framework_support(framework: str) -> bool:
    """Check the given framework is supported or not.

    Args:
        framework (str): A framework name (pytorch or tensorflow)

    Returns:
        bool: Whether the given framework is supported or not.
    """    
    return framework in FRAMEWORK_LIST

def check_model_support(model_name: str, framework: str) -> bool: 
    """Check the given model signature is supported or not.

    Args:
        model_name (str): A model name
        framework (str): A framework name (pytorch or tensorflow)

    Returns:
        bool: Whether the given model signature is supported or not.
    """    
    return model_name in MODEL_LIST and framework in MODEL_LIST[model_name]

def check_optim_support(optim_name: str, framework: str) -> bool:
    """Check the given optimizer signature is supported or not.

    Args:
        optim_name (str): A optimizer name
        framework (str): A framework name (pytorch or tensorflow)

    Returns:
        bool: Whether the given optimizer signature is supported or not.
    """    
    return optim_name in OPTIM_LIST and framework in OPTIM_LIST[optim_name]

def get_model(model_name: str, 
              framework: str,
              pretrained: bool = True) -> Union[torch.nn.Module, tensorflow.keras.Model]:
    """Get model from the given model signature.

    Args:
        model_name (str): A model name
        framework (str): A framework name (pytorch or tensorflow)
        pretrained (bool, optional): Whether it loads weights pretrained with imagenet or not. Defaults to True.

    Returns:
        Union[torch.nn.Module, tensorflow.keras.Model]: The model object from the target framework.
    """    

    if framework == "pytorch":
        return MODEL_LIST[model_name]['pytorch'](pretrained=pretrained)
    elif framework == "tensorflow":
        return MODEL_LIST[model_name]['tensorflow'](weights='imagenet' if pretrained else None)

def get_optim(optim_name: str,
              framework: str,
              learning_rate: float = 0.1) -> Union[torch.optim.Optimizer, tensorflow.keras.optimizers.Optimizer]:
    """[summary]

    Args:
        optim_name (str): An optimizer name
        framework (str): A framework name (pytorch or tenosorflow)
        learning_rate (float, optional): A learning rate. Defaults to 0.1.

    Returns:
        Union[torch.optim.Optimizer, tensorflow.keras.optimizers.Optimizer]: The optimizer object from the target framework.
    """    
    if framework == 'pytorch':
        return OPTIM_LIST[optim_name]['pytorch'](lr=learning_rate)
    elif framework == 'tensorflow':
        return OPTIM_LIST[optim_name]['tensorflow'](learning_rate=learning_rate)
    