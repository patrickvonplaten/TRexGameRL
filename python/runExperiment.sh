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
mkdir ${setupPath}/log
mkdir ${setupPath}/models
cd ${setupPath}

cp ${curPath}/main.py ${setupPath} 
sed -i "/CUR_PATH = os.*/d" ${setupPath}/main.py
sed -i "/PATH_TO_TREX_MODULES =*/c\PATH_TO_TREX_MODULES = \'\/u\/platen\/TRexGameRL\/python\/src\'" ${setupPath}/main.py
sed -i "/PATH_TO_MODELS =*/c\PATH_TO_MODELS = \'${setupPath}\/models\'" ${setupPath}/main.py
sed -i "/PATH_TO_LOG =*/c\PATH_TO_LOG = \'${setupPath}\/log\'" ${setupPath}/main.py
cp ${curPath}/doit.sh ${setupPath}

cd ${setupPath}
#qint doit.sh
#qint -t blabla.1 doit.sh tRexTraining
./main.py

