#!/bin/bash

export workDir=$HOME/Condor_monox;
cd $workDir;
COUNT=0;

for VARLIST in `/bin/cat $workDir/list_variables.txt | cut -d' ' -f1` ; do
    COUNT=$(($COUNT + 1))
    #if [ $COUNT -gt 279 ]; then
    bsub -q 1nd -o $workDir/test_mva/train_${COUNT}.out -J train_${COUNT} $HOME/Condor_monox/run_mva.csh $VARLIST $COUNT
    #fi
done
