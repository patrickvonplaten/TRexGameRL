#!/usr/bin/env bash

cur_path=$(pwd)
log_file=${1}
title=${2}
save_folder=${cur_path}
python_plotter=${cur_path}/../src/tRexPlotter.py

if [ -z "${title}" ]
  then
    title="trial"
fi

python ${python_plotter} --log=${log_file} --save=${save_folder} --title=${title}
