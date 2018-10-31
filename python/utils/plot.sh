#!/usr/bin/env bash

curPath=$(pwd)
experimentsFile=${curPath}/experiments
trialNumToPlot=${1}
saveFolder=${curPath}/plots
pythonPlotter=${curPath}/src/tRexPlotter.py
logFile=$(ls ${experimentsFile}/trial${trialNumToPlot}_*/log/train_log.txt)

if [ -z "${title}" ]
  then
    title="trial${trialNumToPlot}"
fi

python ${pythonPlotter} --log=${logFile} --title=${title} --save=${saveFolder}
echo "Created plots from ${logFile}"
