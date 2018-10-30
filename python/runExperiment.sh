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
setupPathRelativ=experiments/${nameOfExperiment}
mkdir ${setupPath}
mkdir ${setupPath}/log
mkdir ${setupPath}/models
cd ${setupPath}

echo "${HOSTNAME}" > started_on_host.txt
cp ${curPath}/training.config ${setupPath} 
sed -i "/PATH_TO_MODELS=*/c\PATH_TO_MODELS=${setupPathRelativ}\/models\/" ${setupPath}/training.config
sed -i "/PATH_TO_LOG=*/c\PATH_TO_LOG=${setupPathRelativ}\/log\/" ${setupPath}/training.config
cd ${setupPath}
python ${mainPath}

