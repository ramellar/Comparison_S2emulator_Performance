#!/bin/bash

# 1. Setup CMS environment and Proxy
# Note: Condor usually needs the proxy file passed explicitly
export X509_USER_PROXY=$1
export HOME=$PWD 
export XRD_NETWORKSTACK=IPv4
export XRD_STREAMTIMEOUT=300
export XRD_REQUESTTIMEOUT=300

# 2. Setup Conda (adjust the path to where your miniconda is installed)
source /home/llr/cms/pivato/bin/miniconda3/etc/profile.d/conda.sh
conda activate s2_emulator

# 3. Add current directory to PYTHONPATH so it finds 'data_handling'
export PYTHONPATH=$PYTHONPATH:.

cd /home/llr/cms/pivato/Comparison_S2emulator_Performance

# 4. Run the command passed from the sub file
echo "Running: python3 scripts.load_data ${@}"
python3 -m scripts.load_data ${@} 
