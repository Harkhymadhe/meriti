#!/bin/sh

nsys profile --capture-range=cudaProfilerApi \
    --trace='cuda,cudnn,cublas,osrt,opengl,nvtx' \
    --force-overwrite=true --stats=false\
    --output=~/nsys_profiles/report \
    --wait=primary --sample=none --stop-on-exit=true\
    --delay=10 --pytorch=autograd-shapes-nvtx \
    --backtrace=none python3 src/main_nvtx2.py
    # --python-sampling=true --duration=10 --backtrace=none python3 src/main.py


NUM_REPORTS=$(ls -1 ~/nsys_profiles/reports | wc -l)
ID=$($RANDOM | md5sum | head -c 20)

mkdir -p ~/nsys_profiles/reports/report-$NUM_REPORTS-$ID
mv ~/nsys_profiles/report.nsys-rep ~/nsys_profiles/reports/report-$NUM_REPORTS-$ID/report.nsys-rep
# mv ~/nsys_profiles/report.sqlite ~/nsys_profiles/reports/report-$NUM_REPORTS-$ID/report.sqlite

rm ~/nsys_profiles/*.Identifier || echo
