#!/usr/bin/env bash

nameOfExperiment=${1}

if [ $# -eq 0 ]
  then
    echo "Please provide the experiment (name of folder in /src) that you want to delete" 
  exit
fi

if [ ${nameOfExperiment} = "all" ]
  then
	nameOfExperiment=*
fi

curPath=$(pwd)
experimentToDelete=${curPath}/experiments/${nameOfExperiment}
rm -r ${experimentToDelete}
