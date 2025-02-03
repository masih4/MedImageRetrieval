# set all hyper parameters
opts = {}
opts['resize'] = 32
opts['top_k'] = 5
opts['data_path'] = '../dataset/28/adrenalmnist3d.npz'
opts['pretrained_network_name'] = 'DenseNet121'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'
# 'biomedclip'
# 'medclip'
# 'UNI'
# 'openclip'
# 'conch'
# 'virchow'
# 'virchow2'
opts['tsne'] = False
opts['tsne_title'] = 'BreastMNIST'

opts['save_train_hard'] = '../dataset/28/adrenalmnist3d_28/train/'
opts['save_test_hard'] = '../dataset/28/adrenalmnist3d_28/test/'
opts['save_val_hard'] = '../dataset/28/adrenalmnist3d_28/val/'
opts['save_figures'] = '../results/figures/'
opts['save_figures_tsen'] = '../results/figures/tsne/'


opts['CNN'] = True
opts['bath_size'] = 128
