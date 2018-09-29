#!/usr/bin/env bash

nameOfExperiment=${1}

if [ $# -eq 0 ]
  then
    echo "Please provide your experiments name which will be the name of your folder"
  exit
fi

curPath=$(pwd)
setupPath=${curPath}/experiments/${nameOfExperiment}
mkdir ${setupPath}
cd ${setupPath}

cp ${curPath}/main.py ${setupPath} 
cp ${curPath}/doit.sh ${setupPath}

cd ${setupPath}
#qint doit.sh
qint -t blabla.1 doit.sh tRexTraining
