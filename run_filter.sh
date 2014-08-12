#!/bin/bash

source $HOME/EVAL_SH65 5_3_14;

export filterMVAFile=filter_monox.C+;

for dataset in `cat list_samples_analysis.txt | cut -d' ' -f1 ` ; do
  export datasetOLD=$dataset".root";
  echo "filling MVA information in sample: " $dataset;
  root -l -q -b ${filterMVAFile}\(\"${datasetOLD}\"\);
  export datasetNEW=$dataset"_filter.root";
  mv $datasetNEW $datasetOLD;
done
