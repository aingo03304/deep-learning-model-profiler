import argparse
from dlmp import util
from dlmp.exception import (
    UnsupportedModelError,
    InvalidArgumentError
)
from dlmp.util import (
    check_model_sign,
    check_model_support,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=False)
    parser.add_argument('--framework', type=str, required=False)
    parser.add_argument('--model_sign', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--max_iter', type=int, required=False, default=50)
    parser.add_argument('--model_file', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=False)
    args = parser.parse_args()
    
    if not check_model_sign(args.model_name, args.framework, args.model_sign):
        raise InvalidArgumentError('Arguments given is invalid. Please check model_name, framework, or model_sign')
    if not check_model_support(args.model_name + ":" + args.framework):
        raise UnsupportedModelError(args.model_name + ':' + args.framework + ' is unsupported. Check supported model list.')