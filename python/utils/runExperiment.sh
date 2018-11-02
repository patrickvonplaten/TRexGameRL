#!/usr/bin/env bash

nameOfExperiment=${1}

if [ $# -eq 0 ]
  then
    echo "Please provide your experiments name which will be the name of your folder"
  exit
fi

curPath=$(pwd)
basePath=${curPath}
mainPath=${basePath}/main.py
currentHighestNumber=$(ls ${basePath}/experiments | sed 's/\([0-9]\+\).*/\1/g' | awk -F"trial" '{print $2}' | sort -k1.2 | tail -1)
nameOfExperiment="trial$((currentHighestNumber + 1))_${nameOfExperiment}"
setupPath=${basePath}/experiments/${nameOfExperiment}
setupPathRelativ=experiments/${nameOfExperiment}
pythonPath='/u/platen/virtualenvironment/tRex/bin/python2'
mkdir ${setupPath}
mkdir ${setupPath}/log
mkdir ${setupPath}/models
cd ${setupPath}

echo "${HOSTNAME}" > started_on_host.txt
cp ${basePath}/training.config ${setupPath} 
sed -i "/PATH_TO_MODELS=*/c\PATH_TO_MODELS=${setupPathRelativ}\/models\/" ${setupPath}/training.config
sed -i "/PATH_TO_LOG=*/c\PATH_TO_LOG=${setupPathRelativ}\/log\/" ${setupPath}/training.config
cd ${setupPath}
${pythonPath} ${mainPath}

