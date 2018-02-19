#!/bin/bash

# Remove the .DS_Store and ._* files that transfer over from Mac
# Place this file in the parent data directory, and it will recursively remove
# those pesky files from the parent and all subdirectories

find . \( -name '.DS_Store' -or -name '._*' \) -delete