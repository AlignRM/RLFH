from typing import Optional

import pylcs


def count_lines(file_name):
    import subprocess
    out = subprocess.getoutput(f"wc -l {file_name}")
    return int(out.split()[0])


def div(a, b, default_for_zero: Optional[float] = 0):
    return a / b if b != 0 else default_for_zero


def mean(l, default_for_zero=1):
    return div(sum(l), len(l), default_for_zero)


def lcs_sequence_idx(from_str, to_str):
    index = pylcs.lcs_sequence_idx(from_str, to_str)
    valid_index = [i for i, ind in enumerate(index) if ind != -1]
    for i in range(1, len(valid_index)):
        l_bound, r_bound = index[valid_index[i - 1]], index[valid_index[i]]
        char = to_str[r_bound]
        index[valid_index[i]] = to_str.find(char, l_bound + 1, r_bound + 1)
    return index
