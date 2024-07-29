# set all hyper parameters
opts = {}
opts['resize'] = 224
opts['top_k'] = 5
opts['data_path'] = '../dataset/224/pathmnist_224.npz'
opts['pretrained_network_name'] = 'virchow'
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

opts['save_train_hard'] = '../dataset/224/pathmnist_224/train/'
opts['save_test_hard'] = '../dataset/224/pathmnist_224/test/'
opts['save_figures'] = '../results/figures/'


opts['CNN'] = False
opts['bath_size'] = 128
