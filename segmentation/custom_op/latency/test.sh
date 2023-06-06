#!/bin/bash

echo CPU Test
build/dnnl_prof test_cfg.txt office_cpu.csv
sleep 0.5
echo Done

echo GPU Test
build/cudnn_prof test_cfg.txt office_gpu.csv
sleep 0.5
echo Done
