from filter_basic import main as filter_basic
from locate_cells import main as locate_cells
from thresholding import main as thresholding
from os import remove
import uuid

# Using parse arg to input parameters seems to interfere with lower level argparse calls
PATH = r"D:\\kerryn_tiffs\\tiff_dec_sorted"
CLEAR = False

suffix = str(uuid.uuid4())

located_path = "data\\interim\\located_" + "6ed6fa2e-fdbe-4959-be77-8c2ea0d8a0e5"
print("\nlocating cells")
print("=================================================================")
#locate_cells([PATH, "-o", located_path])
print("=================================================================")

filtered_path = "data\\interim\\filtered_" + suffix 
print("\nPerforming basic filtering")
print("=================================================================")
filter_basic([located_path, "-o", filtered_path])
print("=================================================================")

thresholded_path = "data\\processed\\thresholded_" + suffix 
print("\nPerforming thresholding into classes")
print("=================================================================")
thresholding([filtered_path, "-o", thresholded_path])
print("=================================================================")


if CLEAR:
    print("\nRemoving intermediary files")
    remove(located_path)
    remove(filtered_path)

