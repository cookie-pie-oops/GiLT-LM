#!/bin/bash
srun -G4 -c2 -t 1-00:00:00 python train_bllip_con.py 1> >(tee train_out_20250331.log >&1) 2> >(tee train_err.log >&2) 