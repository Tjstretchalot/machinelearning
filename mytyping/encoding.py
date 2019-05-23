"""Criterions for this section as well as stop detectors"""

import torch
import typing

ROUND_DELAYS = True
DELAY_POWERS = 0 # 0 = autoencoder
INPUT_DIM = 29
OUTPUT_DIM = INPUT_DIM + DELAY_POWERS

def round_delay(delay: float) -> int:
    """Rounds the delay to the next larger power of 2, capping it at the maximum delay the
    network can store"""
    if DELAY_POWERS == 0:
        return 0
    if delay <= 0:
        return 0
    if delay >= (1 << (DELAY_POWERS - 1)):
        return 1 << (DELAY_POWERS - 1)
    return 1 << int(delay - 1).bit_length()

def torch_to_tuple(tensor: torch.tensor) -> typing.Tuple[float]:
    """Converts a 1D tensor to a tuple"""
    return tuple(float(a.item()) for a in tensor)

def encode_input(character: str) -> torch.tensor:
    """Converts the specified character to the corresponding tensor that
    should be presented to the network."""

    result = torch.zeros(INPUT_DIM, dtype=torch.double)
    if 'a' <= character <= 'z':
        result[ord(character) - ord('a') + 1] = 1
    elif character == ' ':
        result[27] = 1
    elif character == '\'':
        result[28] = 1
    return torch_to_tuple(result)

def encode_input_stop() -> torch.tensor:
    """Embeds the input stop vector"""
    result = torch.zeros(INPUT_DIM, dtype=torch.double)
    result[0] = 1
    return torch_to_tuple(result)

def get_char_from_index(char_ind: int) -> str:
    """Converts the encoded index into the corresponding character"""
    if char_ind < 26:
        return chr(ord('a') + char_ind)
    elif char_ind == 26:
        return ' '
    else:
        return '\''
    raise ValueError(f'bad index {char_ind}')

def read_input(inp: torch.tensor) -> typing.Tuple[bool, typing.Optional[str]]:
    """Determines what was encoded in the given input. Returns True, char if it corresponds
    with a character and False, None if its a stop signal"""
    if not torch.is_tensor(inp):
        inp = torch.tensor(inp, dtype=torch.double) # pylint: disable=not-callable
    if inp[0] > 0.5:
        return False, None

    return True, get_char_from_index(inp[1:].argmax())

def read_output(output: torch.tensor) -> typing.Tuple[bool, typing.Optional[str], typing.Optional[int]]: # pylint: disable=line-too-long
    """Converts the specified output to the character it represents. Returns
    True, str, delay if the output corresponds to a charecter. Returns False, None, None
    if the output is a stop signal"""

    if not torch.is_tensor(output):
        output = torch.tensor(output, dtype=torch.double) # pylint: disable=not-callable

    # Index 0 is for the stop signal, indices 1-8 are for the delay, index 9-34 are for the character
    if output[0] > 0.5:
        return False, None, None

    delay = 0
    if DELAY_POWERS > 0:
        delay_powers = output[1:1 + DELAY_POWERS] > 0.5
        for i in range(DELAY_POWERS):
            if delay_powers[i]:
                delay += 2 ** i

    char_ind = output[1 + DELAY_POWERS:].argmax()
    character = get_char_from_index(char_ind)

    return True, character, delay

def stop_failer(output: torch.tensor) -> bool:
    """Returns False if the output is encoding a stop signal and True otherwise"""
    return output[0].item() < 0.5

def encode_output(character: str, delay: float) -> torch.tensor:
    """Encodes the given character and delay as an output tensor"""
    if ROUND_DELAYS:
        delay = round_delay(delay)
    result = torch.zeros(OUTPUT_DIM, dtype=torch.double)
    if DELAY_POWERS > 0:
        remdelay = delay
        i = DELAY_POWERS - 1
        while remdelay > 0 and i >= 0:
            if remdelay >= 2 ** i:
                result[i + 1] = 1
                remdelay -= 2 ** i
            i -= 1

    result[1 + DELAY_POWERS:] = torch.tensor(encode_input(character)[1:], dtype=torch.double) # pylint: disable=not-callable
    return torch_to_tuple(result)

def encode_output_stop() -> torch.tensor:
    """Encodes the stop signal"""
    result = torch.zeros(OUTPUT_DIM, dtype=torch.double)
    result[0] = 1
    return torch_to_tuple(result)

def accuracy(actual: 'Sequence', expected: 'Sequence') -> float:
    """Determines the accuracy of the actual sequence when compared to the expected
    sequence"""

    elements_correct = 0
    elements_total = len(actual.raw)
    for ind in range(min(len(expected.raw), len(actual.raw))):
        act = actual.raw[ind]
        exp = expected.raw[ind]

        act_stop, act_char, act_delay = read_output(act)
        exp_stop, exp_char, exp_delay = read_output(exp)

        if act_stop != exp_stop:
            continue
        if act_char != exp_char:
            continue
        if act_delay != exp_delay:
            continue
        elements_correct += 1

    return elements_correct / elements_total

def output_sequence_as_str(seq: 'Sequence') -> str:
    """Converts the output sequence to a string"""
    result = []
    for val in seq.raw:
        is_char, character, _ = read_output(val)
        if is_char:
            result.append(character)
    return ''.join(result)