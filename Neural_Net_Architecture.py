# Two simple functions for applying convulutional neural networks (CNN) and artifical/feed-forward neural networks (ANN/FFN)

# defineNN: define ANN or CNN architecture
def defineNN(x_train, y_train, layer_num, network_seed, ridge_penalty1=0., lasso_penalty1=0., dropout=0, hidden=20, output_shape=3, act_fun='relu'):
    
    # ANN or CNN?
    
    # ANN
    if typo == 'ann':
        
        # set-up inputs bases on input shape
        input1 = tf.keras.Input(shape=x_train.shape[1]*x_train.shape[2])
        
        # normalize input data using its standard deviation
        normalizer = tf.keras.layers.Normalization(axis=(-1))
        normalizer.adapt(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        layers = normalizer(input1)
      
        # define dropout if applied
        if dropout > 0.:
            layers = tf.keras.layers.Dropout(rate=dropout, seed=network_seed)(layers)
      
        # apply each dense layer (1 layer specified)
        for ll in range(1):
          layers = tf.keras.layers.Dense(hidden[0],
                                         activation=act_fun,
                                         use_bias=True,
                                         kernel_regularizer=regularizers.l1_l2(l1=lasso_penalty1, l2=ridge_penalty1),
                                         bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                         kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
                                         )(layers)
      
      
        # OUTPUT layer w/ softmax
        output_layer = tf.keras.layers.Dense(output_shape,
                                             activation=tf.keras.activations.softmax,
                                             # activation = act_fun,
                                             use_bias=True,
                                             kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=0.0),
                                             bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                             kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
                                             )(layers)
        
    # CNN
    elif typo == 'cnn':
        
        # set-up inputs bases on input shape
        input1 = tf.keras.Input(shape=x_train.shape[1:])
        
        # normalize input data using its standard deviation
        normalizer = tf.keras.layers.Normalization(axis=(-1))
        normalizer.adapt(x_train.reshape(x_train.shape[1:]))
        layers = normalizer(input1)      
        
        # convolutional neural network (Three Conv2D layers)
        layers = tf.keras.layers.Conv2D(32, (3, 3),
                                        use_bias=True,
                                        activation=act_fun,
                                        padding="same",
                                        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(input1)
        layers = tf.keras.layers.MaxPooling2D((2, 2))(layers)
        
        layers = tf.keras.layers.Conv2D(32, (3, 3),
                                        use_bias=True,
                                        activation=act_fun,
                                        padding="same",
                                        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(layers)
        layers = tf.keras.layers.MaxPooling2D((2, 2))(layers)
        
        layers = tf.keras.layers.Conv2D(32, (3, 3),
                                        use_bias=True,
                                        activation=act_fun,
                                        padding="same",
                                        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(layers)
        layers = tf.keras.layers.MaxPooling2D((2, 2))(layers) 

        # make final dense layers
        layers = tf.keras.layers.Flatten()(layers)
        layers = tf.keras.layers.Dense(32,
                       activation=act_fun,
                       use_bias=True,
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(layers)
        
        layers = tf.keras.layers.Flatten()(layers)
        layers = tf.keras.layers.Dense(32,
                       activation=act_fun,
                       use_bias=True,
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(layers)


        # DEFINE THE OUTPUT LAYER
        output_layer = tf.keras.layers.Dense(3,
                                             activation=tf.keras.activations.softmax,
                                             use_bias=True,
                                             bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                                             kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(layers)
        
    # CONSTRUCT THE MODEL
    model = tf.keras.Model(input1, output_layer)
    
    return model

  
#############################################################################################################################  
  
  
# make_model: compiles network after using the defineNN function
def make_model(x_train, y_train, typo, npseed, RIDGE1, DROPOUT, nodes, LR):
    
    # DEFINE MODEL
    tf.keras.backend.clear_session()
    model = defineNN(x_train,
                     y_train,
                     typo,
                     network_seed=npseed,
                     ridge_penalty1=RIDGE1,
                     dropout=DROPOUT,
                     hidden=[nodes],
                     act_fun='relu')
    
    # DEFINE LOSS
    if typo == 'ann':
        loss_function = tf.keras.losses.CategoricalCrossentropy()
    elif typo == 'cnn':
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    # COMPILE MODEL
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=loss_function,
                  metrics=[
                      # tf.keras.metrics.SparseCategoricalAccuracy(
                      # name="sparse_categorical_accuracy", dtype=None),
                      tf.keras.metrics.CategoricalAccuracy(
                          name="categorical_accuracy", dtype=None),
                      PredictionAccuracy(y_train.shape[0]),
                      CategoricalTruePositives()])
    
    return model, loss_function
