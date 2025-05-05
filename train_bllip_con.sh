#!/bin/bash
srun -G2 -c8 --mem=1M -t 3-00:00:00 python train_bllip_con.py 1> >(tee train_out_20250426.log >&1) 2> >(tee train_err_20250426.log >&2) 

srun -G2 -c8 -t 8-00:00:00 python train_bllip_ablation_attach.py 1> >(tee train_out_20250416.log >&1) 2> >(tee train_err_20250416.log >&2) 

srun -G1 -c8 --mem=1M -t 8-00:00:00 python train_bllip_ablation_attach.py 1> >(tee train_out_20250425_2.log >&1) 2> >(tee train_err_20250425_2.log >&2) 