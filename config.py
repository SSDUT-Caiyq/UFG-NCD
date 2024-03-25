import os


# Ultra-Fine-Grained Datasets roots
# TODO change datasets root
Root = "/your_ugvc_dataset_root"
SoyAgeingRoot = os.path.join(Root, "SoyAgeing")
UltraRoot = {
    "SoyAgeing-R1": os.path.join(SoyAgeingRoot, "SoyAgeing-R1/R1"),
    "SoyAgeing-R3": os.path.join(SoyAgeingRoot, "SoyAgeing-R3/R3"),
    "SoyAgeing-R4": os.path.join(SoyAgeingRoot, "SoyAgeing-R4/R4"),
    "SoyAgeing-R5": os.path.join(SoyAgeingRoot, "SoyAgeing-R5/R5"),
    "SoyAgeing-R6": os.path.join(SoyAgeingRoot, "SoyAgeing-R6/R6"),
    "SoyGene": os.path.join(Root, "SoyGene/soybeangene"),
    "SoyGlobal": os.path.join(Root, "SoyGlobal/soybean2000"),
    "SoyLocal": os.path.join(Root, "SoyLocal/soybean200"),
    "Cotton": os.path.join(Root, "Cotton80/COTTON"),
}


# Data augment configs
image_size = 448
crop_pct = 0.875
interpolation = 3
trans_type = "imagenet"
n_views = 2

# Model configs
num_proxy_base = 1  # number of base proxies per classes
num_proxy_hard = None  # number of hard proxies per classes
num_proxy_local = 0
mlp_out_dim = 256  # MLP head project dimension
sk_num_iter = 3  # number of Sinkhorn-knopp iters
sk_epsilon = 0.05  # epsilon of Sinkhorn-knopp

# Dataset config
prop_train_labels = 0.5

# Experiment config
temperature = 0.1

# DINO
dino_pretrain_path = "./your_dino_pretrained_path"

# OSR Split dir
osr_split_dir = "/your_osr_split_root"

cifar_10_root = "/your_cifar10_root"
cifar_100_root = "/your_cifar100_root"
cub_root = "/your_cub_root"
aircraft_root = "/your_aircraft_root"
scars_root = "/your_scars_root"
