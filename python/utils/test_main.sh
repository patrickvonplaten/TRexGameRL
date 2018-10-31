#!/usr/bin/env bash

curPath=$(pwd)
${curPath}/./main.py --debug
rm ${curPath}/models/*.epoch.000000*
echo "Cleared dummy ${curPath}/models ..."
