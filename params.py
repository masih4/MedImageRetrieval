# set all hyper parameters
opts = {}
opts['resize'] = 224
opts['top_k'] = 5
opts['data_path'] = '../dataset/224/retinamnist_224.npz'
opts['pretrained_network_name'] = 'EfficientNetV2M'
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

opts['save_train_hard'] = '../dataset/224/retinamnist_224/train/'
opts['save_test_hard'] = '../dataset/224/retinamnist_224/test/'
opts['save_val_hard'] = '../dataset/224/retinamnist_224/val/'
opts['save_figures'] = '../results/figures/'
opts['save_figures_tsen'] = '../results/figures/tsne/'


opts['CNN'] = False
opts['bath_size'] = 128
