# this script remembers to delete file test_arsenic_fit.log
# it is not strictly needed
# usage: $ bash run_tests_with_coverage.sh tests/

rm mrfitty/tests/test_arsenic_fit.log
pytest $1 -s --cov=mrfitty mrfitty/tests/