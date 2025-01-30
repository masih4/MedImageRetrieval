# set all hyper parameters
opts = {}
opts['resize'] = 128
opts['top_k'] = 5
opts['data_path'] = '../dataset/128/bloodmnist_128.npz'
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

opts['save_train_hard'] = '../dataset/128/bloodmnist_128/train/'
opts['save_test_hard'] = '../dataset/128/bloodmnist_128/test/'
opts['save_val_hard'] = '../dataset/128/bloodmnist_128/val/'
opts['save_figures'] = '../results/figures/'
opts['save_figures_tsen'] = '../results/figures/tsne/'


opts['CNN'] = False
opts['bath_size'] = 128
