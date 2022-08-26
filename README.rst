LS Factor Implementations: Comparison of InVEST vs SAGA
=======================================================

This repository is to test out the differences between InVEST's LS factor
implementation and SAGA's

This `main` branch was pushed from the branch
`bugfix/915-ls-factor-different-from-saga-gis` on
https://github.com/phargogh/invest.

To run this script and generate the various outputs in the CWD:

    $ docker run --rm -ti -v$(pwd):/invest -w /invest ghcr.io/phargogh/invest-ls-factor-vs-saga@sha256:bbf9cb863b4f81dda686c0169634226672d7dd3badf585f55e5b9ca040a603bf /bin/bash run-saga-on-gura.sh

