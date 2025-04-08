
#%% Converting a pretrained bert model from Google format to be compatible with Huggingface 
from transformers import AutoModelForPreTraining,AutoTokenizer
"""
To overcome the erors, add  
"""
model_path = "/speakingTrajectories/KAFYNewVersionDecember/pretrainingBert/training_with_jakarta_data/saved_models/model_h3_10KSteps_10resolution/GoogleFormat/bestCheckpoint"
model_path2 = "/speakingTrajectories/KAFYNewVersionDecember/pretrainingBert/training_with_jakarta_data/saved_models/model_h3_10KSteps_10resolution/HuggingFaceFormat"

# Load the model with from_tf=True
model = AutoModelForPreTraining.from_pretrained(model_path, from_tf=True)
model.save_pretrained(model_path2, saved_model_name="pytorch_model.bin")

tokenizer = AutoTokenizer.from_pretrained(model_path, from_tf=True)
tokenizer.save_pretrained(model_path2)
#%%
# Trying m,asking with the new pretrained model
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

traj  ="8a8c1078ea6ffff 8a8c1078ea67fff 8a8c1078eb4ffff 8a8c1078eb47fff 8a8c1078eb67fff 8a8c10419b67fff 8a8c1078c4c7fff 8a8c1078c4f7fff 8a8c1078c4e7fff 8a8c1078c4c7fff 8a8c1078c4cffff 8a8c104ce3a7fff 8a8c1078eb6ffff 8a8c1078eb4ffff 8a8c1078ea67fff 8a8c1078ea6ffff 8a8c1068e69ffff 8a8c10788da7fff"
# 8a8c12b6a947fff
# traj  = "8a8c10795657fff 8a8c1068390ffff 8a8c12b694d7fff 8a8c106eaa8ffff"
file_path = "/speakingTrajectories/KAFYNewVersionDecember/pretrainingBert/training_with_jakarta_data/saved_models/model_h3_10KSteps_10resolution/HuggingFaceFormat"
model = AutoModelForMaskedLM.from_pretrained(file_path)
tokenizer = AutoTokenizer.from_pretrained(file_path)
inputs = tokenizer(traj, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(logits)
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
#%%
import folium
import h3

# Input H3 indices as a string
h3_indices_string = "8a8c1078ea6ffff 8a8c1078ea67fff 8a8c1078eb4ffff 8a8c1078eb47fff 8a8c1078eb67fff 8a8c10419b67fff 8a8c1078c4c7fff 8a8c1078c4f7fff 8a8c1078c4e7fff 8a8c1078c4c7fff 8a8c1078c4cffff 8a8c104ce3a7fff 8a8c1078eb6ffff 8a8c1078eb4ffff 8a8c1078ea67fff 8a8c1078ea6ffff 8a8c1068e69ffff 8a8c10788da7fff"
# h3_indices_string = "8a8c1078ea6ffff 8a8c1078ea67fff 8a8c1078eb4ffff 8a8c1078eb47fff 8a8c1078eb67fff"
# h3_indices_string = "8a8c1078ea6ffff 8a8c1078ea67fff 8a8c100e0597fff 8a8c1078eb47fff 8a8c1078eb67fff"


# Split the string into a list of H3 indices
h3_indices = h3_indices_string.split()

# Detokenize: Convert H3 indices to latitude and longitude
coordinates = [h3.cell_to_latlng(index,) for index in h3_indices]

# Create a folium map centered at the mean of the coordinates
latitudes, longitudes = zip(*coordinates)
map_center = [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)]
m = folium.Map(location=map_center, zoom_start=12)

# Add a LineString (PolyLine) connecting the coordinates
folium.PolyLine(
    locations=coordinates,
    color="blue",
    weight=5,
    opacity=0.8,
).add_to(m)

# Display the map
m
#%%
import os
import sys
sys.path.append(os.path.abspath('/speakingTrajectories/KAFYNewVersionDecember/'))
from KAFY import KAFY
# from KAFY import KA
args = {
    "KAFY_LLM": 'BERT',
    "KAFY_args":{""
        "project_path": "/speakingTrajectories/KAFYNewVersionDecember/projectExample",
         "detokenizer": "token2point_cluster_centroid",
               "beam_size": 10,
        "beam_normalization": 0.7,
        "use_constraints": True
        }
}
kafy = KAFY(**args['KAFY_args'])
import pandas as pd
from shapely.geometry import Point

# Specify the path to your GeoDataFrame pickle file
file_path = '/speakingTrajectories/datasets/jakarta/train_small_10trajs.GeoDataFrame.pickle'

# Load the GeoDataFrame
test_gdf = pd.read_pickle(file_path)
# %% Imputation Testing
first_trajectory = test_gdf['geometry'].iloc[0] #This is a MultiPoint object

# Convert MultiPoint geometry to List[Point]
points_list = list(first_trajectory.geoms)  # Access points using .geoms
# for point in 
print((((points_list[0]))))
# Find the middle index of the list
mid_index = len(points_list) // 2
# This is the original point a the middle of the trajectory
print("The point just before the middle\n",points_list[mid_index-1])
print("This is the original point a the middle of the trajectory\n",points_list[mid_index])
print("The point just after the middle\n",points_list[mid_index+1])

# # Divide the list into two parts
# first_half = points_list[:mid_index]
# second_half = points_list[mid_index:]
# Remove the middle point from the list
points_list_without_middle = points_list[:mid_index] + points_list[mid_index + 1:]
result = kafy.impute_a_gap(points_list_without_middle,mid_index)
print(result)


# %%
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString

# Original trajectory points
first_trajectory = test_gdf['geometry'].iloc[0]  # This is a MultiPoint object
points_list = list(first_trajectory.geoms)  # Convert MultiPoint geometry to List[Point]

# Find the middle index
mid_index = (len(points_list) // 2)+10

# Remove the middle point and get the imputed point
points_list_without_middle = points_list[:mid_index] + points_list[mid_index + 1:]
result = kafy.impute_a_gap(points_list_without_middle, mid_index)
imputed_point = result['imputed_seg_pt'][0]  # Predicted point from the model

# Create a LineString for the original trajectory
trajectory_line = LineString(points_list)

# Plot the trajectory
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the original trajectory in blue
x, y = trajectory_line.xy
ax.plot(x, y, color='blue', linewidth=2, label="Original Trajectory")

# Highlight the middle point (removed) in red
middle_point = points_list[mid_index]
ax.scatter(middle_point.x, middle_point.y, color='red', s=100, label="Original Middle Point", zorder=5)

# Highlight the predicted point in green
ax.scatter(imputed_point.x, imputed_point.y, color='green', s=100, label="Predicted Middle Point", zorder=5)

# Add labels and legend
ax.set_title("Trajectory with Predicted Middle Point")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

# Show the plot
plt.show()
# %%
import folium
from shapely.geometry import Point, LineString

# Original trajectory points
first_trajectory = test_gdf['geometry'].iloc[0]  # This is a MultiPoint object
points_list = list(first_trajectory.geoms)  # Convert MultiPoint geometry to List[Point]

# Find the middle index
mid_index = (len(points_list) // 2)+10

# Remove the middle point and get the imputed point
points_list_without_middle = points_list[:mid_index] + points_list[mid_index + 1:]
result = kafy.impute_a_gap(points_list_without_middle, mid_index)
imputed_point = result['imputed_seg_pt'][0]  # Predicted point from the model

# Get coordinates for the trajectory
trajectory_coords = [(point.y, point.x) for point in points_list]  # Convert to (lat, lon)
predicted_point_coords = (imputed_point.y, imputed_point.x)  # Predicted point (lat, lon)
middle_point_coords = (points_list[mid_index].y, points_list[mid_index].x)  # Original middle point (lat, lon)

# Create a Folium map centered at the trajectory's centroid
map_center = [first_trajectory.centroid.y, first_trajectory.centroid.x]
m = folium.Map(location=map_center, zoom_start=15)

# Plot the original trajectory as a blue polyline
folium.PolyLine(trajectory_coords, color="blue", weight=2.5, opacity=1, tooltip="Original Trajectory").add_to(m)

# Add a red marker for the original middle point
folium.Marker(
    location=middle_point_coords,
    icon=folium.Icon(color="red", icon="info-sign"),
    popup="Original Point"
).add_to(m)

# Add a green marker for the predicted middle point
folium.Marker(
    location=predicted_point_coords,
    icon=folium.Icon(color="green", icon="info-sign"),
    popup="Imputed Point"
).add_to(m)

# Display the map
m

#%%
import os
import sys
sys.path.append(os.path.abspath('/speakingTrajectories/KAFYNewVersionDecember/'))

from KAFY import KAFY
args = {
    "KAFY_LLM": 'BERT',
    "KAFY_args":{
        "project_path": "/speakingTrajectories/KAFYNewVersionDecember/projectExample",
       
        }
}
kafy = KAFY(**args['KAFY_args'])
#Test on 10 trajectories data from Porto dataset
import pandas as pd
from shapely.geometry import Point

# Specify the path to your GeoDataFrame pickle file
file_path = '/speakingTrajectories/datasets/jakarta/train_small_10trajs.GeoDataFrame.pickle'

# Load the GeoDataFrame
test_gdf = pd.read_pickle(file_path)

# %% Prediction Testing
first_trajectory = test_gdf['geometry'].iloc[0] #This is a MultiPoint object

# Convert MultiPoint geometry to List[Point]
points_list = list(first_trajectory.geoms)  # Access points using .geoms
# for point in 
print((((points_list[0]))))
# Find the middle index of the list
mid_index = len(points_list) // 2

# Divide the list into two parts
first_half = points_list[:mid_index]
second_half = points_list[mid_index:]

result = kafy.predict_is_next_trajectory(first_half,second_half)
if result==1:
    print("Yes this is the next trajectory")
else:
    print("No this is not the next trajectory")



#%%
# # Get the first 10 trajecotries of this big file
# import pandas as pd

# # Specify the path to your GeoDataFrame pickle file
# file_path = '/speakingTrajectories/datasets/jakarta/train.GeoDataFrame.pickle'

# # Load the GeoDataFrame
# test_gdf = pd.read_pickle(file_path)

# # Extract the first 10 rows
# small_gdf = test_gdf.head(10)

# # Specify the path to save the smaller GeoDataFrame
# output_file_path = '/speakingTrajectories/datasets/jakarta/train_small_10trajs.GeoDataFrame.pickle'

# # Save the smaller GeoDataFrame
# small_gdf.to_pickle(output_file_path)

# print(f"Smaller GeoDataFrame saved to {output_file_path}")

# %%
#Plot pretraining results as a loss graph .csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/speakingTrajectories/KAFYNewVersionDecember/pretrainingBert/training_with_jakarta_data/saved_models/model_h3_30KSteps_10resolution_all_data/GoogleFormat/training_loss.csv'
output_image_path = '/speakingTrajectories/KAFYNewVersionDecember/pretrainingBert/training_with_jakarta_data/saved_models/model_h3_30KSteps_10resolution_all_data/GoogleFormat/training_loss.png'


df = pd.read_csv(file_path)
# Smoothing function using exponential moving average (EMA)
def smooth_values(values, smoothing_factor=0.8):
    smoothed = []
    last = values[0]  # Initialize with the first value
    for value in values:
        smoothed_value = last * smoothing_factor + (1 - smoothing_factor) * value
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed

# Apply smoothing to the 'Value' column
df['Smoothed_Value'] = smooth_values(df['Value'], smoothing_factor=0.6)

# Set a professional Seaborn style
sns.set_theme(style="whitegrid")

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(df['Step'], df['Value'], color="steelblue", linewidth=1, alpha=0.5, label="Original Loss (No Smoothing)")
plt.plot(df['Step'], df['Smoothed_Value'], color="darkorange", linewidth=2.5, label="Smoothed Loss")

# Add labels and title
plt.title("Smoothed Loss vs Step for BERT Pretraining (Jakarta resolution=10, ALL Data)", fontsize=16, weight='bold')
plt.xlabel("Training Step", fontsize=14)
plt.ylabel("Loss", fontsize=14)

# Customize ticks and grid
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

# Add a legend
plt.legend(fontsize=12, loc="upper right", frameon=True, shadow=True, borderpad=1)

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig(output_image_path, dpi=1200, bbox_inches="tight")
# %% Tying ti import GPT-2 my pretrained using nanoGPT into HF

from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("/speakingTrajectories/Transformers/nanoGPT/out-trajectory-FullJakartaPlusSimpleSummaryJune2024")
model = GPT2Model.from_pretrained("/speakingTrajectories/Transformers/nanoGPT/out-trajectory-FullJakartaPlusSimpleSummaryJune2024")
model
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

# %%
