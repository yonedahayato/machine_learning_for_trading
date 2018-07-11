#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# install library
# docker build -f ./dockerfile/Dockerfile_lib -t rl_lib .

# pytest
# env HELPER_DIR=$SCRIPT_DIR/helper docker-compose run rl pytest ./unit_test/unit_test.py -v

# train and validate
env HELPER_DIR=$SCRIPT_DIR/helper docker-compose run rl python Q_Learning_with_Tables_and_NN.py

docker-compose down
