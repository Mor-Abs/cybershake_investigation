# list of stations to plot
# st_list = ["HNPS", "TRMS"]
st_list = ["32002e1", "CHHC"]
# st_list = ["PIPS"]
# st_list = ["CHHC"]

# import  dependencies
import os
import sys
from pathlib import Path
import importlib

# change working directory to recognize modules
# .py
file_dir = Path(__file__).resolve().parent
root_dir = file_dir.parent
# # .ipynb
# file_dir = os.getcwd()
# root_dir = os.path.dirname(file_dir)

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import utils.util as util
importlib.reload(util)
from utils.util import plot_specific_stations


# define pathes
plots_saving_dir = file_dir / "plots"
os.makedirs(plots_saving_dir, exist_ok=True)


# call the function to generate a map
fig, st_df = plot_specific_stations(
    st_list=st_list,
    desired_width_cm=8.5,
    intended_dpi=900,
    file_name="stations_map_selected.png",
    plots_saving_dir=Path(plots_saving_dir),
    plot_using_mapdata=False,
)
