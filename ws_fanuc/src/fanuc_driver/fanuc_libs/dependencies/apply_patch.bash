#!/bin/bash

if [[ -z $(git status --porcelain) ]]; then
  git apply "$1"
fi
