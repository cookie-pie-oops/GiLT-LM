#!/bin/bash
srun -G1 -c8 -t 1-00:00:00 python train_bllip_con.py --debug 1> >(tee train_out_20250331.log >&1) 2> >(tee train_err_20250331.log >&2) 