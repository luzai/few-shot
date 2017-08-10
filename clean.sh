#!/usr/bin/env bash
rm output* -rf
find . -name '*.pyc' -exec rm {} \;
find . -name '*.pdf' -exec rm {} \;
find . -name '*.png' -exec rm {} \;
find . -name '*.log' -exec rm {} \;
