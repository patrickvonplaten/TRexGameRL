#!/usr/bin/env bash

nameOfExperiment=${1}

if [ $# -eq 0 ]
  then
    echo "Please provide your experiments name which will be the name of your folder"
  exit
fi

curPath=$(pwd)
mainPath=${curPath}/main.py
setupPath=${curPath}/experiments/${nameOfExperiment}
mkdir ${setupPath}
mkdir ${setupPath}/log
mkdir ${setupPath}/models
cd ${setupPath}

echo "${HOSTNAME}" > started_on_host.txt
cp ${curPath}/training.config ${setupPath} 
cd ${setupPath}
python ${mainPath}

