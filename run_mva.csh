#!/bin/tcsh

set VARLIST=$1;
set COUNT=$2;

source $HOME/EVAL65 5_3_14;
set workDir=$HOME/Condor_monox;
cd $workDir;

set trainMVA_File=trainMVA_monox.C+;
set NSELS=0;
set NSELB=3;
set SIG_TRAIN=/afs/cern.ch/work/c/ceballos/mva/samples/ttlj.root;
set BKG_TRAIN=/afs/cern.ch/work/c/ceballos/mva/samples/zll.root;
set TAG=default_${COUNT};
set METHODS=BDTG;

root -l -q -b $HOME/Condor_monox/${trainMVA_File}\(${NSELS},${NSELB},\"${SIG_TRAIN}\",\"${BKG_TRAIN}\",\"${TAG}\",\"${METHODS}\",\"${VARLIST}\"\);
rm -f default_${COUNT}.root;

set evaluateMVAFile=evaluateMVA_monox.C+;
foreach dataset (`cat list_samples.txt | cut -d' ' -f1 `)
    set datasetOLD=$dataset".root"
    echo "filling MVA information in sample: "  $dataset
    root -l -q -b $HOME/Condor_monox/${evaluateMVAFile}\(\"${datasetOLD}\",\"${METHODS}\",\"${VARLIST}\",\"${TAG}\",\"weights/\",\"${VARLIST}\",0,${COUNT}\);
end
