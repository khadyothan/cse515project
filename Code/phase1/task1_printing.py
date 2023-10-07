def print_cm_grid(grid_data, grid_index):
    print(f"Grid {grid_index + 1} (R, G, B channels):")
    for channel_index, channel_name in enumerate(["R", "G", "B"]):
        mean, sd, skewness = grid_data[channel_index]
        print(f"  | {channel_name}: Mean={mean:.2f}, SD={sd:.2f}, Skew={skewness:.2f} |", end="")
    print()
    
def print_hog_grid(grid_data, grid_index):
    print(f"Grid {grid_index + 1} (HOG Histogram Values):")
    for i, values in enumerate(grid_data):
        print(f"  | Bin {i + 1}: {values} |")   
    print()
      

def readable_output(data, input_feature_model):
    if input_feature_model == "color_moments":
        for i, grid_row in enumerate(data):
            for j, grid in enumerate(grid_row):
                print_cm_grid(grid, grid_index=i * 10 + j)
    elif input_feature_model == "hog":
        for i, grid_row in enumerate(data):
            for j, grid in enumerate(grid_row):
                print_hog_grid(grid, grid_index=i * 10 + j)
    else:
        print(data)