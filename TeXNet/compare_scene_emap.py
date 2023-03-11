import os
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR = "/home/gautamsree/Downloads/new_HADAR_database" # Root directory of the data
if "sreekum1" in os.getcwd():
    DATA_DIR = "/research/hal-sreekum1/HADAR_Fanglin" # Root directory of the data
else:
    DATA_DIR = "/home/gautamsreekumar/research/hal-sreekum1/HADAR_Fanglin" # Root directory of the data
SCENES_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Scenes to check
OUT_DIR = "heatcube_emap_visualization"
FRAME_NUMS = [1, 2, 3, 4, 5] # number of frames to check
SIDES = ["L", "R"] # left and right sides

# matnames_data = scio.loadmat("matName_full.mat")["tmp"]
matnames_data = scio.loadmat("matName_FullDatabase.mat")["matName"]
matnames = [_[0][0] for _ in matnames_data]

print("Found materials")

for i, name in enumerate(matnames):
    print(i, name)

os.makedirs(OUT_DIR, exist_ok=True)

def pseudo_color(S):
    c = S.shape[2]
    r = S[..., 0]
    g = S[..., c//2]
    b = S[..., -1]
    r = (r-np.min(r))/(np.max(r)-np.min(r))
    g = (g-np.min(g))/(np.max(g)-np.min(g))
    b = (b-np.min(b))/(np.max(b)-np.min(b))
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

for scene_id in SCENES_IDS:
    print("Checking scene", scene_id)
    scene_folder = os.path.join(DATA_DIR, "Scene{}".format(scene_id))
    groundtruth_folder = os.path.join(scene_folder, "GroundTruth")
    eMap_folder = os.path.join(groundtruth_folder, "eMap")
    heatcube_folder = os.path.join(scene_folder, "HeatCubes")

    if scene_id == 11:
        FRAME_NUMS = [1, 2, 3, 4]
    else:
        FRAME_NUMS = [1, 2, 3, 4, 5]

    for frame in FRAME_NUMS:
        for side in SIDES:
            print("Checking frame", frame, side)
            S_file = os.path.join(heatcube_folder, f"{side}_{frame:04d}_heatcube.mat")
            if "S" in scio.loadmat(S_file).keys():
                S = scio.loadmat(S_file)["S"]
            elif "HSI" in scio.loadmat(S_file).keys():
                S = scio.loadmat(S_file)["HSI"]
            else:
                raise ValueError("No known keys found in heatcube file")
            e_file = os.path.join(eMap_folder, f"new_eMap_{side}_{frame:04d}.npy")
            e = np.load(e_file)

            mats_in_scene = np.unique(e)
            matnames_in_scene = [matnames[i] for i in mats_in_scene]

            S_col = pseudo_color(S)

            fig, ax = plt.subplots(1, 3, figsize=(10, 5))
            ax[0].imshow(S_col)
            ax[0].set_title("Heatcube (pseudo-color")

            im = ax[1].imshow(e, cmap="jet")
            ax[1].set_title("eMap")
            cmap = plt.cm.jet  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            bounds = np.linspace(0, 30, 31)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)
            cb = mpl.colorbar.ColorbarBase(ax[2], cmap=cmap, norm=norm,
                                            spacing='proportional', ticks=mats_in_scene, boundaries=bounds, format='%1i')
            
            cb.ax.set_yticklabels(matnames_in_scene)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"scene{scene_id}_{side}_{frame:04d}.png"))

            plt.close()
            plt.clf()
