#!/bin/bash
srun -G2 -c8 --mem 1M python train_bllip_con.py 1> >(tee train_out.log >&1) 2> >(tee train_err.log >&2)