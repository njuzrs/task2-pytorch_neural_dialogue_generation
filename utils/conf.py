class disc_config(object):
    vocab_size = 25003
    train_dir = './disc_data/'
    maxlen = 50
    batch_size = 256
    embed_dim = 512
    num_layers = 2
    max_len = 50
    grad_clip = 5.0
    lr = 0.001
    disc_steps = 5
    
    '''
    batch_size = 256
    lr = 0.2
    lr_decay = 0.9
    vocab_size = 25000
    embed_dim = 512
    steps_per_checkpoint = 200
    #hidden_neural_size = 128
    num_layers = 2
    train_dir = './disc_data/'
    name_model = "disc_model"
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./disc_data/batch_piece/"
    #query_len = 0
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.5
    #num_epoch = 60
    #max_decay_epoch = 30
    max_grad_norm = 5
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    '''


class gen_config(object):
    train_dir = "./gen_data/"
    vocab_size = 25003
    emb_dim = 512
    hidden_size = 512
    lr = 0.01
    maxlen = 50
    batch_size = 128
    grad_clip = 5.0
    max_train_data_size = None
    gen_steps = 1
    '''
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 128
    emb_dim = 512
    num_layers = 2
    vocab_size = 25000
    train_dir = "./gen_data/"
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]
    '''





