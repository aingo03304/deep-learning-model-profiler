"""
Exception
"""

class DLMPException(Exception):
    pass

class UnsupportedModelError(DLMPException):
    pass

class UnsupportedFrameworkError(DLMPException):
    pass

class InvalidArgumentError(DLMPException):
    pass