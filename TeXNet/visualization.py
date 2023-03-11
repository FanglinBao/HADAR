import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import sys
from tqdm import trange
import scipy.special as scsp
from skimage import img_as_float
from skimage import exposure
import torchmetrics

# Give data location (folder name) in DATA_DIR and
# visualization results in OUT_DIR

if len(sys.argv) == 1:
    DATA_DIR = 'supervised'
else:
    DATA_DIR = sys.argv[1]
OUT_DIR = DATA_DIR+'_visualized'

os.makedirs(OUT_DIR, exist_ok=True)

max_n_class = 30
T_max = 70

def get_X_from_V(S, v):
    C, H, W = S.shape
    S1 = (np.mean(S[:, :H//2]))
    S2 = (np.mean(S[:, H//2:]))

    # print(v[0])

    X = v[0]*S1 + v[1]*S1

    return X

def visualize_TeX(TeX, fname, max_vals, kind='pred'):
    T_max = max_vals[0]
    S_max = max_vals[1]
    assert len(TeX.shape) == 3
    assert kind in ['pred', 'gt', 'residue']
    if kind == 'pred':
        kind = 'Prediction'
    elif kind == 'gt':
        kind = 'GT'
    else:
        kind = "Residue"

    # Need to normalize each value of TeX to [0, 1]
    TeX[..., 0] /= max_n_class # Divide e-map by the number of classes
    TeX[..., 1] /= T_max # Divide the T-map by the maximum temperature.
    TeX[..., 2] /= S_max # divide by the maximum of np.mean(S_half) among each half.

    TeX_ = mpl.colors.hsv_to_rgb(TeX)/np.amax(mpl.colors.hsv_to_rgb(TeX))
    TeX_ = ((TeX_ - np.min(TeX_))/(np.max(TeX_) - np.min(TeX_)*255.).astype(np.uint8))
    plt.imshow(TeX_)
    plt.title('TeX '+kind)
    plt.axis('off')
    # plt.colorbar()
    plt.clim(0,1)
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_residue(i, j, res, fname):
    # Need to normalize each value of TeX to [0, 1]

    img = np.squeeze(np.linalg.norm(res, axis=0))
    # np.save('/home/sureshbs/Desktop/TeXNet/Dataset/TexNet_multiscene/'+OUT_DIR+'/'+f'residue_{i+j}_no_rescale.npy', img)
    np.save(OUT_DIR+'/'+f'residue_{i+j}_no_rescale.npy', img)
    img = img_as_float(np.squeeze(np.linalg.norm(res, axis=0)))/255.0
    # print(np.max(img))

    img_adapteq = exposure.equalize_adapthist(img_as_float(img), clip_limit=0.2)

    # np.save(f'residue_{i}_no_rescale.npy', img)
    plt.imshow(img_adapteq)
    plt.title('residue')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0,1)
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def visualize_T(data, fname, error=False, kind='pred'):
    assert len(data.shape) == 2
    assert kind in ['pred', 'gt']
    kind = 'Prediction' if kind == 'pred' else 'GT'

    mu = 15.997467357494212
    std = 8.544861474951992

    # mu = np.load('/home/sureshbs/Desktop/TeXNet/Dataset/HADAR_database/Scene14/GroundTruth/tMap/T_mu.npy')
    # std = np.load('/home/sureshbs/Desktop/TeXNet/Dataset/HADAR_database/Scene14/GroundTruth/tMap/T_std.npy')

    # denormalize temperature data, if we are not plotting error
    if not error:
        data = data*std + mu
    else:
        data = data*std


    plt.imshow(data, cmap='turbo', vmin=0, vmax=70)
    if not error:
        plt.title('Temperature '+kind)
    else:
        plt.title('L1 error in temperature prediction')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_m_CE(data, fname):
    assert len(data.shape) == 2

    # Even though CE is unbounded, it will not be higher than 4 unless
    # it is a very wrong prediction, at which point we are not concerned
    # about how bad it is.
    plt.imshow(data, cmap='turbo', vmin=0, vmax=2)
    plt.title('Cross entropy error in e-map prediction')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def visualize_m(data, fname, kind='pred'):
    assert kind in ['pred', 'gt']
    kind = 'Prediction' if kind == 'pred' else 'GT'

    # hue = np.array([[0.5569],
    #                [0.5529],
    #                [0.3098],
    #                [0.8588],
    #                [0.4196],
    #                [0.2   ],
    #                [0.902 ],
    #                [0.5833],
    #                [0.1386],
    #                [0.8   ],
    #                [0.78  ],
    #                [0.76  ],
    #                [0.74  ],
    #                [0.72  ],
    #                [0.7   ],
    #                [0.95  ],
    #                [0.93  ],
    #                [0.91  ],
    #                [0.451 ],
    #                [0.41  ],
    #                [0.6627],
    #                [0.1   ],
    #                [0.1586],
    #                [0.04  ],
    #                [0.2641],
    #                [0.97  ],
    #                [0.5693],
    #                [0.    ],
    #                ])
    hue = np.array([[0.5098],
                    [0.6157],
                    [0.8784],
                    [0.0431],
                    [0.0745],
                    [0.1059],
                    [0.3451],
                    [0.0196],
                    [0.1176],
                    [0.5059],
                    [0.0588],
                    [0.0941],
                    [0.5961],
                    [0.5255],
                    [0.0784],
                    [0.0039],
                    [0.5373],
                    [0.1294],
                    [0.4510],
                    [0.1529],
                    [0.1843],
                    [0.4784],
                    [0.0706],
                    [0.0392],
                    [0.3137],
                    [0.8706],
                    [0.5882],
                    [0.9373],
                    [0.0392],
                    [0.1961]])

    sat = np.ones_like(hue)*0.7
    val = np.ones_like(hue)

    hsv = np.concatenate((hue, sat, val), 1)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    rgb = np.concatenate((rgb, np.ones((max_n_class, 1))), 1)

    newcmap = ListedColormap(rgb)

    mycmap = plt.get_cmap('gist_rainbow', max_n_class)

    plt.imshow(data, cmap=mycmap, vmin=0, vmax=max_n_class-1)
    plt.title('Material map '+kind)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks([])
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()

    return

def visualize_m_error(data, fname):
    mycmap = plt.get_cmap('gray', 2)
    plt.imshow(data, cmap=mycmap)
    plt.title('Error in material map')
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['correct', 'wrong'])
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    return

def save_m_file(data, fname):
    np.save(os.path.join(OUT_DIR, fname), data)


def visualize_v(data, fname, kind='pred'):
    assert kind in ['pred', 'gt', 'l1error', 'kldiv']

    vmax = 1.
    if kind == 'gt':
        kind = 'GT'
    elif kind == 'pred':
        kind = 'Prediction'
    elif kind == 'l1error':
        kind = 'L1 error'
    elif kind == 'kldiv':
        kind = 'KL-div'
        vmax = 4

    if kind != 'KL-div':
        v1 = np.squeeze(data[0, ...])
        v2 = np.squeeze(data[1, ...])
        #v3 = np.squeeze(data[2, ...])
        #v4 = np.squeeze(data[3, ...])

        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(v1, vmin=0., vmax=vmax, cmap='turbo')
        ax[0].axis('off')
        ax[0].set_title('v1')
        plt.colorbar(im, ax=ax[0])

        im = ax[1].imshow(v2, vmin=0., vmax=vmax, cmap='turbo')
        ax[1].axis('off')
        ax[1].set_title('v2')
        plt.colorbar(im, ax=ax[1])

        #im = ax[1, 0].imshow(v3, vmin=0., vmax=vmax, cmap='turbo')
        #ax[1, 0].axis('off')
        #ax[1, 0].set_title('v3')
        #plt.colorbar(im, ax=ax[1, 0])

        #im = ax[1, 1].imshow(v4, vmin=0., vmax=vmax, cmap='turbo')
        #ax[1, 1].axis('off')
        #ax[1, 1].set_title('v4')
        #plt.colorbar(im, ax=ax[1, 1])

        plt.suptitle('v-map '+kind)

        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.clf()
        plt.close()
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(data, vmin=0., vmax=vmax, cmap='turbo')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        plt.suptitle('v-map '+kind)

        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.clf()
        plt.close()

    return

# Calculate the metrics for each large and small samples
large_miou = []
small_miou = []
syn_miou = []
exp_miou = []

print("folder: ", DATA_DIR)

for j in range(44):
    T_file = torch.load(os.path.join(DATA_DIR, f'val_T_{j}.pt'), map_location='cpu')
    e_file = torch.load(os.path.join(DATA_DIR, f'val_e_{j}.pt'), map_location='cpu')
    v_file = torch.load(os.path.join(DATA_DIR, f'val_v_{j}.pt'), map_location='cpu')
    pred_file = torch.load(os.path.join(DATA_DIR, f'val_pred_{j}.pt'), map_location='cpu')
    S_pred_file = torch.load(os.path.join(DATA_DIR, f'val_S_pred_{j}.pt'), map_location='cpu')
    S_true_file = torch.load(os.path.join(DATA_DIR, f'val_S_true_{j}.pt'), map_location='cpu')

    # T_file = torch.load(os.path.join(DATA_DIR, 'val_T.pt'), map_location='cpu')
    # e_file = torch.load(os.path.join(DATA_DIR, 'val_e.pt'), map_location='cpu')
    # v_file = torch.load(os.path.join(DATA_DIR, 'val_v.pt'), map_location='cpu')
    # pred_file = torch.load(os.path.join(DATA_DIR, 'val_pred.pt'), map_location='cpu')
    # S_pred_file = torch.load(os.path.join(DATA_DIR, 'val_S_pred.pt'), map_location='cpu')
    # S_true_file = torch.load(os.path.join(DATA_DIR, 'val_S_true.pt'), map_location='cpu')

    assert T_file.size(0) == e_file.size(0) == v_file.size(0) == pred_file.size(0)

    n = T_file.size(0)

    nclass = max_n_class

    for i in trange(n):
        pred = pred_file[i].squeeze()
        e = e_file[i].squeeze()

        T = T_file[i].squeeze().numpy()

        v = v_file[i].squeeze().numpy()

        c = pred.size(0)
        e_pred = pred[:nclass]
        T_pred = None
        v_pred = None
        if c == nclass+1:
            T_pred = pred[nclass]
        elif c == nclass+4:
            v_pred = pred[nclass:]
        elif c == nclass+3:
            T_pred = pred[nclass]
            v_pred = pred[nclass+1:]

            # e_pred = F.softmax(e_pred, 0).squeeze()

        if nclass == max_n_class:
            # e_pred_ = torch.argmax(e_pred, 0, keepdim=False)
            e_pred_ = torch.argmax(e_pred.unsqueeze(0), 1, keepdim=False)
            e_ = e.unsqueeze(0)

            miou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=nclass)
            val = miou(e_pred_, e_).item()

            # append to the right type of metric
            if e.shape[0] == 256:
                small_miou.append(val)
            elif e.shape[0] == 1080:
                large_miou.append(val)
                syn_miou.append(val)
            elif e.shape[0] == 260:
                large_miou.append(val)
                exp_miou.append(val)

            del miou

            e_ce_error = F.cross_entropy(e_pred.unsqueeze(0), e.unsqueeze(0), reduction='none').squeeze()
            e = e.numpy()
            e_pred = torch.argmax(e_pred, 0).squeeze().numpy()
            e_error = (e_pred != e).astype(int)

            visualize_m(e_pred, fname=f'emap_pred_{i+j*n}.png', kind='pred')
            visualize_m(e, fname=f'emap_GT_{i+j*n}.png', kind='gt')
            # print("e error", j, i, np.mean(e_error.astype(float)))
            visualize_m_error(e_error, fname=f'emap_error_{i+j*n}.png')
            visualize_m_CE(e_ce_error, fname=f'emap_CE_error_{i+j*n}.png')
            save_m_file(e_pred, fname=f'm_pred_{i+j*n}.npy')

        if T_pred is not None:
            T_pred = T_pred.squeeze().numpy()
            visualize_T(T_pred, fname=f'Tmap_pred_{i+j*n}.png', error=False, kind='pred')
            visualize_T(T, fname=f'Tmap_GT_{i+j*n}.png', error=False, kind='gt')
            save_m_file(T_pred, fname=f'T_pred_{i+j*n}.npy')
            T_error = np.abs(T-T_pred)
            visualize_T(T_error, fname=f'Tmap_error_{i+j*n}.png', error=True)

        if v_pred is not None:
            v_pred = v_pred.squeeze()
            v = v.squeeze()
            v_pred = F.softmax(v_pred, 0).numpy()
            save_m_file(v_pred, fname=f'v_pred_{i+j*n}.npy')
            v_error = np.abs(v-v_pred)

            v_kldiv = scsp.rel_entr(v, v_pred).sum(0)

            visualize_v(v_pred, fname=f'vmap_pred_{i+j*n}.png', kind='pred')
            visualize_v(v, fname=f'vmap_GT_{i+j*n}.png', kind='gt')
            visualize_v(v_error, fname=f'vmap_error_{i+j*n}.png', kind='l1error')
            visualize_v(v_kldiv, fname=f'vmap_KLdiv_{i+j*n}.png', kind='kldiv')

        # Load S_pred and S_true here.
        S_pred = S_pred_file[i].squeeze().numpy()
        S_true = S_true_file[i].squeeze().numpy()

        X_pred = get_X_from_V(S_pred, v_pred)
        X_true = get_X_from_V(S_true, v)

        # print(np.shape(np.dstack((e_pred, T_pred, X_pred))))

        # TeX_pred = np.concatenate((e_pred, T_pred, X_pred), 2)
        # TeX_true = np.concatenate((e, T, X_true), 2)
        TeX_pred = np.dstack((e_pred, T_pred, X_pred))
        TeX_true = np.dstack((e, T, X_true))
        S_res = (np.log(np.abs(S_true-S_pred)))

        T_max = np.max(np.maximum(T, T_pred))
        C, H, W = S_pred.shape
        S_max1 = np.maximum(np.mean(S_pred[:, :H//2]), np.mean(S_pred[:, H//2:]))
        S_max2 = np.maximum(np.mean(S_true[:, :H//2]), np.mean(S_true[:, H//2:]))
        S_max = np.maximum(S_max1, S_max1)

        visualize_TeX(TeX_pred, fname=f'TeX_pred_{i+j*n}.png', max_vals=[T_max, S_max], kind='pred')
        visualize_TeX(TeX_true, fname=f'TeX_GT_{i+j*n}.png', max_vals=[T_max, S_max], kind='gt')
        visualize_residue(i, j, S_res, fname=f'S_residue_{i+j*n}.png')

print("Average large mIoU", np.mean(large_miou))
print("Average small mIoU", np.mean(small_miou))
print("Average synthetic mIoU", np.mean(syn_miou))
print("Average experimental mIoU", np.mean(exp_miou))
