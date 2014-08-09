#!/bin/bash

export filterMVAFile=filter_monox.C+;

for dataset in `cat list_samples_analysis.txt | cut -d' ' -f1 ` ; do
  export datasetOLD=$dataset"_mva.root";
  echo "filling MVA information in sample: "  $dataset;
  root -l -q -b ${filterMVAFile}\(\"${datasetOLD}\"\);
  #export datasetNEW=$dataset"_mva_filter.root";
  #mv $datasetNEW $datasetOLD;
done
