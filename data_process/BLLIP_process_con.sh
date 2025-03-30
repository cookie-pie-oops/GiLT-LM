#!/bin/bash
python BLLIP_process_con.py 1> >(tee bllip_out.log >&1) 2> >(tee bllip_err.log >&2)