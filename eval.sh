#!/bin/bash
srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_sg_test.py 1> >(tee eval_out_sg.log >&1) 2> >(tee eval_err_sg.log >&2)
srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_marginal.py 1> >(tee eval_out_mar_2.log >&1) 2> >(tee eval_err_mar_2.log >&2) 
srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_joint.py 1> >(tee eval_out_joint.log >&1) 2> >(tee eval_err_joint.log >&2) 