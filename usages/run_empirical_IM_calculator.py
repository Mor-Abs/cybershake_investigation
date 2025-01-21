"""
# Empirical_IM_Calculator.py is cybershake_investigation.gen_emprical_archive

Code Description:
This code is developed to run the Empirical_IM_Calculator.py with predefined arguments.
It calls Empirical_IM_Calculator.py from utils repo.
See the last line to run this code in bash terminal.

Author: Morteza
Version History:
- Version 1.0: November 18, 2024
"""

# Import dependencies
import os
import sys
from pathlib import Path
import importlib

# Change working directory to recognize modules
try:
    # .py execution
    file_dir = Path(__file__).resolve().parent
except NameError:
    # .ipynb execution
    file_dir = Path(os.getcwd())

root_dir = file_dir.parent
sys.path.append(str(root_dir))  # Add root directory to sys.path for imports
os.chdir(root_dir)

# Import and reload the module
import utils.Empirical_IM_Calculator as Empirical_IM_Calculator

importlib.reload(Empirical_IM_Calculator)
from utils.Empirical_IM_Calculator import main

if __name__ == "__main__":
    # Define input arguments as a list
    sys.argv = [
        "Empirical_IM_Calculator.py",
        "/mnt/hypo_data/mab419/Cybershake_Data/v24p9/",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/NZ_FLTmodel_2010.txt",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/meta_config.yaml",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/model_config.yaml",
        "/mnt/hypo_data/mab419/Empirical_Data/v24p9/",
        "--ss_db",
        "/mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/new_flt_site_source.db",
        "--component",
        "geom",
        "--n_procs",
        "20",
    ]

    # Run the main function from Empirical_IM_Calculator.py
    main()


########## To run Empirical_IM_Calculator.py from terminal use the following code in bash:
# python /mnt/hypo_data/mab419/mab419_cybershake_investigation/utils/Empirical_IM_Calculator.py /mnt/hypo_data/mab419/Cybershake_Data/combined_v20p6_v24p8/ /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30 /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/NZ_FLTmodel_2010.txt /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/meta_config.yaml /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/model_config.yaml /mnt/hypo_data/mab419/Empirical_Data/combined_v20p6_v24p8/ --ss_db /mnt/hypo_data/mab419/mab419_cybershake_investigation/base_data/new_flt_site_source.db --component geom --n_procs 30
