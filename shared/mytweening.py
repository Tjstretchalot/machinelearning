"""Additional tweening functions not in pytweening
"""

def doublespeed(n): # pylint: disable=invalid-name
    """A non-symmetric easing which moves linearly from 0 to 1 in first 0.5
    and then stays constant for last 0.5
    """

    return n * 2 if n < 0.5 else 1

def smoothstep(n): # pylint: disable=invalid-name
    """A symmetric easing that starts and ends with first derivative 0

    Args:
        n (float): a value between 0 and 1

    Returns:
        n (float): a smoothed value between 0 and 1
    """

    return n * n * (3.0 - 2.0 * n)

def smootheststep(n): #pylint: disable=invalid-name
    """A symmetric easing that starts and ends with first, second, and third derivative 0

    Args:
        n (float): a value between 0 and 1

    Returns:
        n (float): a smoothed value between 0 and 1
    """
    return n * n * n * n * (35 + n * (-84 + n * (70 + n * -20)))