import torch
import tensorflow

from dlmp.util import (
    check_model_sign,
    check_model_support,
    get_model_by_sign
)

def test_check_model_sign():
    