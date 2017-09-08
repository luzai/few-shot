#!/usr/bin/env bash

find -name "*.pdf" -exec rename 's/_/-/g' {} ";"
find -name "*.pdf" -exec rename 's/0\.0/0-0/g' {} ";"
