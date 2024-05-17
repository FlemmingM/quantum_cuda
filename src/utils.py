'''
Python util functions for data analysis
'''

import numpy as np


def decimal_to_binary(num: int) -> str:
    """Function to convert decimals to binary

    :param num: an integer
    :type num: int
    :return: the corresponding binary string
    :rtype: str
    """
    return str(bin(num))[2:]


def pad_binary(binary_str: str, length: int) -> str:
    """Function to pad binary strings with 0s in the front
    to allow for same lenth of strings.

    :param binary_str: binary string
    :type binary_str: str
    :param length: the total length the string should have.
    :type length: int
    :return: the binary string padded to the length.
    :rtype: str
    """
    return '0'*(length-len(binary_str)) + binary_str