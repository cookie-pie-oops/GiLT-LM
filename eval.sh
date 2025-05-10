#!/bin/bash
srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_sg_test.py --run_name push_bllip_con_test_gas1 1> >(tee eval_out_sg_2.log >&1) 2> >(tee eval_err_sg_2.log >&2)

srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_blimp.py --run_name push_bllip_con_test_gas1_s150 1> >(tee eval_out_blimp_s150.log >&1) 2> >(tee eval_err_blimp_s150.log >&2)

srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_marginal.py --run_name push_bllip_con_test_gas1 1> >(tee eval_out_mar_2.log >&1) 2> >(tee eval_err_mar_2.log >&2) 

srun -G1 -c8 --mem=1M -t 8-00:00:00 python eval_joint.py --run_name push_bllip_con_test_gas1 1> >(tee eval_out_joint.log >&1) 2> >(tee eval_err_joint.log >&2) 