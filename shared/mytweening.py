"""Additional tweening functions not in pytweening
"""

def smoothstep(n): # pylint: disable=invalid-name
    """A symmetric easing that starts and ends with first derivative 0

    Args:
        n (float): a value between 0 and 1

    Returns:
        n (float): a smoothed value between 0 and 1
    """

    return n * n * (3.0 - 2.0 * n)
