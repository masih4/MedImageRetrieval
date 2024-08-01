# set all hyper parameters
opts = {}
opts['resize'] = 32
opts['top_k'] = 5
opts['data_path'] = '../dataset/28/breastmnist.npz'
opts['pretrained_network_name'] = 'medclip'
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

opts['save_train_hard'] = '../dataset/224/bloodmnist_224/train/'
opts['save_test_hard'] = '../dataset/224/bloodmnist_224/test/'
opts['save_figures'] = '../results/figures/'


opts['CNN'] = False
opts['bath_size'] = 32
