#!/usr/bin/env bash

numOfExperiment=${1}
curPath=$(pwd)

if [ $# -eq 0 ]
  then
    echo "Please provide the experiment (name of folder in /src) that you want to delete" 
  exit
fi

if [ ${numOfExperiment} = "all" ]
  then
	experimentToDelete=${curPath}/experiments/*
else
	experimentToDelete=${curPath}/experiments/trial${numOfExperiment}_*
fi
rm -r ${experimentToDelete}
