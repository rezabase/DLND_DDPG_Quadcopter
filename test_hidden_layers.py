from keras import layers, models






#Used the following for Test1 - Test4. All saved weights are based on the following model
def my_first_actor(input_layer):
    reg_lambda = 0.01
        
    net = layers.Dense(units=512,kernel_regularizer=layers.regularizers.l2(reg_lambda))(input_layer)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
        
    net = layers.Dense(units=512,kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    #net = layers.Dense(units=512)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
        
    net = layers.Dense(units=512,kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    #net = layers.Dense(units=256)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
    return net



#Used the following for Test1 - Test4. All saved weights are based on the following model
def my_first_critic(input_layer):
    reg_lambda = 0.01
    
    net = layers.Dense(units=400,kernel_regularizer=layers.regularizers.l2(reg_lambda))(input_layer)
    #net = layers.Dense(units=512)(input_layer)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
        
    net = layers.Dense(units=400, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    #net = layers.Dense(units=512)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
        
    net = layers.Dense(units=200, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    #net = layers.Dense(units=256)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
    return net






'''
def small(input_layer):
    reg_lambda = 0.001
    
    #net = layers.Dense(units=128,kernel_regularizer=layers.regularizers.l2(reg_lambda))(input_layer)
    net = layers.Dense(units=64)(input_layer)
    #net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.1)(net)
        
    net = layers.Dense(units=128, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    #net = layers.Dense(units=128)(net)
    #net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    #net = layers.Dropout(0.1)(net)
    return net
'''
def small(input_layer):
    net = layers.Dense(units=64, activation='relu')(input_layer)
    net = layers.Dropout(0.04)(net)
        
    net = layers.Dense(units=128, activation='relu')(input_layer)
    net = layers.Dropout(0.04)(net)

    net = layers.Dense(units=64, activation='relu')(input_layer)
    net = layers.Dropout(0.04)(net)
    return net


def medium(input_layer):
    reg_lambda = 0.001
    
    #net = layers.Dense(units=256,kernel_regularizer=layers.regularizers.l2(reg_lambda))(input_layer)
    net = layers.Dense(units=256)(input_layer)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.1)(net)
        
    #net = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    net = layers.Dense(units=512)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.1)(net)
        
    #net = layers.Dense(units=512, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    net = layers.Dense(units=512)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.1)(net)
    return net





#Used the following for Test5
def large(input_layer):
    reg_lambda = 0.01
    net = layers.Dense(units=800,kernel_regularizer=layers.regularizers.l2(reg_lambda))(input_layer)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
        
    net = layers.Dense(units=800, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
        
    net = layers.Dense(units=800, kernel_regularizer=layers.regularizers.l2(reg_lambda))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)
    return net

