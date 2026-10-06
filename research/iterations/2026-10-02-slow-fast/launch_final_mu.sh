cd ~/WunderFunder-creative; S=research/iterations/2026-10-02-slow-fast/final_mu.sh; chmod +x $S
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
L(){ setsid nohup $S "$@" > /dev/null 2>&1 < /dev/null & }
L fmu256_s3 3 0 d256; L fmu256_s4 4 1 d256; L fmu128_s3 3 2 h128; L fmu128_s4 4 4 h128; L fmuc500_s3 3 5 c500
