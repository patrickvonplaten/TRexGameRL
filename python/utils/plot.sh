#!/usr/bin/env bash

curPath=$(pwd)
experimentsFile=${curPath}/experiments
logFile=${1}
saveFolder=${curPath}/plots
pythonPlotter=${curPath}/src/tRexPlotter.py

if [ -z "${title}" ]
  then
    title="trial"
fi

python ${pythonPlotter} --log=${logFile} --save=${saveFolder} --title=${title}
