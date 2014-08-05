#!/bin/bash

source $HOME/EVAL_SH65 5_3_14;
export workDir=$HOME/Condor_monox;
cd $workDir;

for dataset in `cat list_samples.txt | cut -d' ' -f1 ` ; do
    export datasetOLD=${dataset}".root"
    export datasetNEW=${dataset}"_mva.root"
    echo "filling MVA information in sample: "  ${dataset}
    cp ${datasetOLD} ${datasetNEW};
    COUNT=0;
    for VARLIST in `/bin/cat $workDir/list_variables.txt | cut -d' ' -f1` ; do
        COUNT=$(($COUNT + 1))
        export datasetAUX=${dataset}"_mva_"${COUNT}".root"
	root -l -q -b $HOME/Condor_monox/merge.C+\(\"${datasetNEW}\",\"${datasetAUX}\",\"bdt_monox_\",\"${VARLIST}\"\);
    done
done
