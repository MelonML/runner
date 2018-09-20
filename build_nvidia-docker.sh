#!/usr/bin/env bash

nvidia-docker build -f Dockerfile.nvidia -t openautoml/runner:0.1-nvidia .