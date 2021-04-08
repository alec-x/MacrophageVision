# Data Flow

locate_cells.py
directory with tiffs -> np.array[1]
*filtered for crop size and singularity*

filter_basic.py
np.array[1] -> np.array[1]

filter_classes.py
np.array[1] -> np.array[2]

array_to_pd.py
np.array[2] -> pd.df[3] + directory_with_images[4]

[1] shape = (num_samples, num_channels, crop_size, crop_size) (pickle)
[2] shape = (num_samples, num_channels, crop_size, crop_size, channel_threshold) (pickle)
[3] row = samples. col = index, img_path, channel_threshold_1...n (pickle)
[4] dir/channel_name/#####channel_name.png (dir/[3] pickle)
