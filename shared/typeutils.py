"""A collection of methods for type checking"""

import numpy as np

def check(**kwargs):
    """Every keyword argument should be a tuple of the form (val, expected_types). The val is the thing
    whose type you want to check, expected_types is either a single type or a tuple of types."""

    for key, kwarg in kwargs.items():
        val, expected_types = kwarg
        if not isinstance(val, expected_types):
            if isinstance(expected_types, tuple):
                joined = ', '.join(expected_types)
                raise ValueError(f'expected {key} is one of {joined} but got {val} (type={type(val)})')
            else:
                raise ValueError(f'expected {key} is {expected_types} but got {val} (type={type(val)})')

def check_list(expected_types, **kwargs):
    """Verifies that val is list-like and contains only the specified types. The val
    should be passed in as a keyword argument so we know its name"""
    if len(kwargs) != 1:
        raise ValueError(f'this requires exactly 1 keyword argument')

    key, val = list(kwargs.items())[0]
    check(**{key: (val, (list, tuple))})

    if len(val) < 100:
        for i, item in enumerate(val):
            check(**{f'{key}[{i}]': (item, expected_types)})
    else:
        indexes = np.random.choice(len(val), 100, replace=False)
        for ind in indexes:
            check(**{f'{key}[{ind}]': (val[ind], expected_types)})

def check_callable(**kwargs):
    """Checks that every keyword argument is callable"""
    for key, val in kwargs.items():
        if not callable(val):
            raise ValueError(f'expected {key} is callable, got {val} (type={type(val)})')

def _make_shape_str(expected_shape) -> str:
    result = []
    for item in expected_shape:
        if item is None:
            result.append('any')
        elif isinstance(item, int):
            result.append(str(item))
        elif isinstance(item, str):
            result.append(f'{item}=any')
        else:
            name, amt = item
            if amt is None:
                result.append(f'{name}=any')
            else:
                result.append(f'{name}={amt}')
    return '(' + ', '.join(result) + ')'

def _check_shape(actual, expected) -> bool:
    if len(actual) != len(expected):
        return False
    for ind, act in enumerate(actual):
        exp = expected[ind]
        if isinstance(exp, int):
            if act != exp:
                return False
        elif isinstance(exp, (tuple, list)):
            if isinstance(exp[1], int) and act != exp[1]:
                return False
    return True

def check_tensors(**kwargs):
    """Every keyword argument should be a tuple in the form (val, expected_shape, expected_dtype).
    Each shape component may be one of:
        int (interpreted as a specific value)
        None (interpreted as any value is ok)
        str (interpreted as any value ok, has name if other issues arise)
        tuple(str, int): interpreted as expect a certain size which has a name

    Example:
        check_tensors(inp=(inp, (30, 15), torch.double))
            ValueError('expected inp has shape (30, 15) but has shape (20, 15)')
            ValueError('expected inp has dtype torch.float64 but has dtype torch.float32')
        check_tensors(inp=(inp, (('batch', None), ('hidden size', 15)), torch.double))
            ValueError('expected inp has shape (batch=any, hidden_size=15) but has shape (20, 10))
    """
    for key, kwarg in kwargs.items():
        val, expected_shape, expected_dtype = kwarg
        if not _check_shape(val.shape, expected_shape):
            raise ValueError(f'expected {key} has shape {_make_shape_str(expected_shape)} '
                             + f'but has shape {str(tuple(val.shape))}')
        if val.dtype != expected_dtype:
            raise ValueError(f'expected {key} has dtype {expected_dtype} but has dtype {val.dtype}')
