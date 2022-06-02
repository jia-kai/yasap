#!/bin/bash -e

src=$1
dst=$2

if [ -z "$dst" ]; then
    echo "usage: $0 <src file> <dst file>"
    exit 1
fi

exec exiftool -TagsFromFile "$src" -overwrite_original \
    "-all:all>all:all" "$dst"
