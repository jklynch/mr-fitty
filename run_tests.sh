#!/bin/bash
# usage: $ run_tests.sh tests/

py.test $1 -s --cov=src/mrfitty