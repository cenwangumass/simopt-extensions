import pytest
import simopt_extensions


def test_sum_as_string():
    assert simopt_extensions.sum_as_string(1, 1) == "2"
