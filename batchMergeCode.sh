#!/bin/bash

export workDir=$HOME/Condor_monox;
cd $workDir;

bsub -q 1nd -o $workDir/test_mva/mergeAll.out -J mergeAll $HOME/Condor_monox/mergeCode.sh
