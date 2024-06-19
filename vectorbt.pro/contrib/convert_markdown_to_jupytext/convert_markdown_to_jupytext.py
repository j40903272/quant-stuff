import pathlib
import re
import sys

def remove_re(pattern, line):
    line = re.sub(pattern + " ", "", line)
    line = re.sub(" " + pattern, "", line)
    line = re.sub(pattern, "", line)
    return line

def convert(input_lines):
    # code is enclosed in "```pycon" and "```"
    inside_code = False

    # info is started from "!!! " and follwed the lines with "   " prefix
    inside_info = False

    yield '''# %%
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# %% [markdown]
"""
'''
    for line in input_lines:
        if inside_code:
            if line == "```\n":
                yield '\n# %% [markdown]\n"""\n'
                inside_code = False
            elif line.startswith(">>> ") or line.startswith("... "):
                yield line[4:]
            else:
                # skip code output
                pass

        elif line == "```pycon\n":
            yield '"""\n\n# %%\n'
            inside_code = True
        else:
            if inside_info:
                if line.startswith("    "):
                    line = f"> {line[4:]}"
                else:
                    inside_info = False

            if line.startswith("!!! "):
                inside_info = True
                line = f"> **{line[4:-1]}**\n>\n"

            # remove emoji
            # :some-emoji:
            line = remove_re(r"\:[\w-]+\:", line)

            # remove svg
            line = remove_re(r"^!\[\]\(/assets/.+?.svg\)", line)
            yield line
    yield '"""\n'


def main(input_file_path):
    """convert vectorbt markdown doc to jupyter notebook in jupytext percent format
    """
    with pathlib.Path(input_file_path).open("r") as input_file:
        print(''.join(convert(input_file)))


if __name__ == "__main__":
    main(sys.argv[1])
