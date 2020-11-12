import argparse
from dlmp import util
from dlmp.exception import (
    UnsupportedModelError
)

def check_model_sign(model_name, framework, model_sign):
    if model_name + ":" + framework == model_sign:
        return True
    elif framework and model_name:
        return True
    elif model_sign:
        return True
    return False

def check_model_support(model_sign): 
    model_name, framework = model_sign.split(":")
    if model_name in util.MODEL_LIST and framework in util.MODEL_LIST[model_name]:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=False)
    parser.add_argument('--framework', type=str, required=False)
    parser.add_argument('--model_sign', type=str, required=False)
    args = parser.parse_args()
    
    if not check_model_sign(args.model_name, args.framework, args.model_sign):
        raise InvalidArgumentError('Arguments given is invalid. Please check model_name, framework, or model_sign')
    if not check_model_support(args.model_name + ":" + args.framework):
        raise UnsupportedModelError(args.model_name + ':' + args.framework + ' is unsupported. Check supported model list.')