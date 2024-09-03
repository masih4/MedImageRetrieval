# set all hyper parameters
opts = {}
opts['resize'] = 64
opts['top_k'] = 5
opts['data_path'] = '../dataset/64/adrenalmnist3d_64.npz'
opts['pretrained_network_name'] = 'UNI'
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

opts['save_train_hard'] = '../dataset/64/adrenalmnist3d_64/train/'
opts['save_test_hard'] = '../dataset/64/adrenalmnist3d_64/test/'
opts['save_figures'] = '../results/figures/'
opts['save_figures_tsen'] = '../results/figures/tsne/'


opts['CNN'] = False
opts['bath_size'] = 32
