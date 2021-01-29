"""
    Deep Learning Model Profiler (DLMP)
    ==========================================

    DLMP profiles all the model provided by TensorFlow and PyTorch.

    :copyright: (c) 2020 Minjae Kim.
    :license: MIT License.
"""

import argparse
from exception import (
    UnsupportedFrameworkError,
    UnsupportedModelError,
    InvalidArgumentError
)
from util import (
    check_model_support,
    check_framework_support,
    get_model,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', type=str, required=True)
    parser.add_argument('-f', '--framework', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=32)
    parser.add_argument('-o', '--offset_iter', type=int, required=False, default=50)
    parser.add_argument('-M', '--max_iter', type=int, required=False, default=50)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-p', '--pretrained', type=bool, required=False, default=True)
    args = parser.parse_args()
    
    if not check_framework_support(args.framework):
        raise UnsupportedFrameworkError(args.framework + ' is not supported.')
    
    if not check_model_support(args.model_name, args.framework):
        raise UnsupportedModelError(args.model_name + ':' + args.framework + ' is unsupported. Check supported model list.')
    