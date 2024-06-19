#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

V=${1:?"Missing version"}
perl -i -pe "s/.*/__version__ = \"$V\"/ if \$.==3" ../vectorbtpro/_version.py
