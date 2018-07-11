#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# install library
# docker build -f ./dockerfile/Dockerfile_lib -t rl_lib .

env HELPER_DIR=$SCRIPT_DIR/helper docker-compose run rl pytest ./unit_test/unit_test.py -v

docker-compose down
