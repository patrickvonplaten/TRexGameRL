#!/usr/bin/env bash

curPath=$(pwd)
${curPath}/./main.py --debug
rm ${curPath}/models/*
