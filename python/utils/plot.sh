#!/usr/bin/env bash

curPath=$(pwd)
experimentsFile=${curPath}/experiments
trialNumToPlot=${1}
saveFolder=${curPath}/plots
pythonPlotter=${curPath}/src/tRexPlotter.py
cd ${experimentsFile}
trialName=$( find . -name trial${trialNumToPlot}_* | sed 's/.\///g')
cd ${curPath}
logFile=${experimentsFile}/${trialName}/log/train_log.txt


if [ -z "${title}" ]
  then
    title=${trialName}
fi

python ${pythonPlotter} --log=${logFile} --title=${title} --save=${saveFolder}
echo "Created plots from ${logFile}"
