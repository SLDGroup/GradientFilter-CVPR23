#!/bin/bash

echo CPU Test
build/dnnl_prof test_cfg.txt cpu.csv
sleep 0.5
echo Done

echo GPU Test
build/cudnn_prof test_cfg.txt gpu.csv
sleep 0.5
echo Done
