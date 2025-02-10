# import  dependencies
import os
import sys
from pathlib import Path
import shutil
import pickle

file_dir=Path(__file__).resolve().parent
root_dir = file_dir.parent

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import utils.seismic_hazard_analysis as sha

# directory work
data_version = "combined_v20p6_v24p8"
sim_dir = f"/mnt/hypo_data/mab419/Cybershake_Data/{data_version}"  # dpath to simulation data
sim_pickle_dir = f"/mnt/hypo_data/mab419/Cybershake_Data_Pickle/{data_version}"  # dpath to hazard calculation data

# Check if the destination directory exists
if Path(sim_pickle_dir).exists():
    prompt = input(
        f"The following path already exists! Do you want to delete and renew (1) or terminate (2)? \n{sim_pickle_dir}\n"
    )
    if prompt == '1':
        # Remove the folder
        shutil.rmtree(sim_pickle_dir)
        print(f"Deleted and renewed the path: \n{sim_pickle_dir}")
        os.makedirs(sim_pickle_dir, exist_ok=False)
    elif prompt == '2':
        print("Terminating the process.")
        exit()
    else:
        print("Invalid input. Terminating the process.")
        exit()

else:
    os.makedirs(sim_pickle_dir, exist_ok=False)
    print(f"Created the path: \n{sim_pickle_dir}")

# Load Data from CSV Files
fault_im_data = sha.nshm_2010.load_sim_im_data(Path(sim_dir)) 

# Write Pickle File
file_name = "Cybershake_fault_im_data.pkl"
file_path = os.path.join(sim_pickle_dir, file_name)
with open(file_path, "wb") as file:
    pickle.dump(fault_im_data, file)

print(f"Dictionary saved to {file_path}")