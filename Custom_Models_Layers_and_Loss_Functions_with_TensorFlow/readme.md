# [Custom Models, Layers, and Loss Functions with TensorFlow](https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow/home/welcome)

---

Welcome to Custom Models, Layers, and Loss Functions with TensorFlow! You’re joining thousands of learners currently enrolled in the course.

I'm excited to have you in the class and look forward to your contributions to the learning community.

To begin, I recommend taking a few minutes to explore the course site. Review the material we’ll cover each week, and preview the assignments you’ll need to complete

to pass the course. Click Discussions to see forums where you can discuss the course material with fellow students taking the class.

If you have questions about course content, please post them in the forums to get help from others in the course community.

For technical problems with the Coursera platform, visit the Learner Help Center.

Good luck as you get started, and I hope you enjoy the course!

## Content

### 1. Week1 -> Funcitional API 
  #### * Create a model Using functional API( Notebook ->  [1st_notebook.ipynb](https://github.com/ANKITPODDER2000/Tensorflow-Advance/blob/main/Custom_Models_Layers_and_Loss_Functions_with_TensorFlow/Week1/1st_notebook.ipynb) )
  ```python
    def create_functionalapi_model():
      input_layer = Input(shape=(28,28))

      #1st way -> i. flatten_layer = Flatten() ii. flatten_layer(input_layer)
      #2nd way -> i. flatten_layer = Flatten()(input_layer)
      flatten_layer = Flatten()(input_layer)

      #1st way -> i. first_layer = Dense(128 , activation=tf.nn.relu) ii. first_layer(flatten_layer)
      #2nd way -> i. first_layer = Dense(128 , activation=tf.nn.relu)(flatten_layer)
      first_layer = Dense(128 , activation=tf.nn.relu)(flatten_layer)

      output_layer = Dense(10 , activation=tf.nn.softmax)(first_layer)
      model = Model(inputs = input_layer , outputs = output_layer)
      model.compile(optimizer = "rmsprop" , loss = tf.keras.losses.categorical_crossentropy , metrics = ['acc'])
      return model
  ```
    
  #### * Create Multi Output Architecture
  ---
  <p align="center">
  <img src="https://user-images.githubusercontent.com/50513363/99687028-33650f00-2aaa-11eb-98ef-0b25b0421412.png" margin="0 auto" display="block"/>
  </p>

  #### * Create Multi Input Architecture / Siamese network
  ---
  <p align="center">
  <img src="https://user-images.githubusercontent.com/50513363/99687391-9eaee100-2aaa-11eb-96b6-f1ac555a7d5d.png" margin="0 auto" display="block"/>
  </p>
  
### 2. Week2 -> Custom Loss Function
  * Huber Loss Function 
  
  ```python
  def huber_loss(y_true , y_pred):
    thresold = 1
    error = y_true - y_pred
    return_type = tf.abs(error) <= thresold
    r1 = 0.5 * tf.square(error)
    r2 = thresold * (tf.abs(error) - (0.5*thresold))
    return tf.where(return_type , r1 , r2)
  ```
  
  * Huber Loss With function *wrapper*
  
  ```python
  
def huber_loss_wrapper(thresold):
    def huber_loss(y_true , y_pred):
        error = y_true - y_pred
        return_type = tf.abs(error) <= thresold
        r1 = 0.5 * tf.square(error)
        r2 = thresold * (tf.abs(error) - (0.5*thresold))
        return tf.where(return_type , r1 , r2)
    return huber_loss
  ```
  
  * Huber Loss class
  
  ```python
  
class Huber(Loss):
    thresold = 1
    def __init__(self , thresold):
        super().__init__()
        self.thresold = thresold
    def call(self , y_true , y_pred):
        error = y_true - y_pred
        return_type = tf.abs(error) <= self.thresold
        r1 = 0.5 * tf.square(error)
        r2 = self.thresold * (tf.abs(error) - (0.5*self.thresold))
        return tf.where(return_type , r1 , r2)
  ```
  

### 3. Week3 -> Custom Layers
  * Introduction to Lambda Layers
  * Custom Functions from Lambda Layers
  ```python
    model_lambda = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28,28)),
      tf.keras.layers.Dense(128),
      tf.keras.layers.Lambda(lambda x : tf.abs(x)),
      tf.keras.layers.Dense(10 , activation='softmax')
  ])

  model_lambda.compile(optimizer = RMSprop() , loss = categorical_crossentropy , metrics = ['acc'])
  model_lambda.summary()
  model_lambda.fit(train_data , train_label , epochs=5)
  model_lambda.evaluate(test_data , test_label)
  ```
  * Architecture of a Custom Layer
  * Coding your own custom Dense Layer
  ```python
  #CUSTOM DENSE LAYER CLASS
  class SimpleDense(Layer):
    def __init__(self , units = 32):
        super(SimpleDense , self).__init__()
        self.units = units

    def build(self , input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name = "kernal" , 
                             initial_value = w_init(shape = (input_shape[-1] , self.units) , dtype="float32") , 
                             trainable=True )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name = "bias",
                             initial_value = b_init(shape = (self.units) , dtype = "float32") ,
                             trainable=True)
    def call(self , inputs):
        return tf.matmul(inputs , self.w) + self.b
  ```
  * Custom Layer with activation 
  ```python

  class MyDenseLayerwithActivation(Layer):
      def __init__(self , units = 32 ,activation = None):
          super(MyDenseLayerwithActivation , self).__init__()
          self.units = units
          self.activation = activation

      def build(self , input_shape):
          w_init = tf.random_normal_initializer()
          b_init = tf.zeros_initializer()

          self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1] , self.units) , dtype="float32") , 
                                trainable=True , name="kernal")
          self.b = tf.Variable(initial_value=b_init(shape=(self.units , ) , dtype="float32") , 
                                trainable=True , name="bias")
      def call(self , inputs):
          return self.activation( tf.matmul(inputs , self.w) + self.b )
  ```

### 4. Week4 -> Custom Models
  * Explain some benefits to defining a custom model class instead of using the Functional or Sequential APIs
  * Define a model by creating a Python class that inherits from TensorFlow's Model class
  * Describe functions that can be inherited from the TensorFlow Model class
  * Build a residual network by defining a custom model class
  #### important lines of code ->
  
  ---
  ---
  ```python
  class IdentityBlock(Model):
    def __init__(self , filters , kernal_size):
        super(IdentityBlock , self ).__init__(name  ='')
        self.conv = Conv2D(filters , kernal_size , padding='same')
        self.norm = BatchNormalization()
        self.act  = Activation('relu')
        self.add  = Add()
    def call(self , input):
        x = self.conv(input)
        x = self.norm(x)
        x = self.act(x)

        x = self.conv(x)
        x = self.norm(x)

        x = self.add([x , input])
        x = self.act(x)
        return x
        
  class Resnet(Model):
    def __init__(self , num_classes , activation = 'softmax'):
        super(Resnet , self).__init__(name="")
        self.conv7 = Conv2D(64 , 7 , padding='same')
        self.norm  = BatchNormalization()
        self.pool  = MaxPool2D((3,3))
        self.idbl1 = IdentityBlock(64 , 3)
        self.idbl2 = IdentityBlock(64 , 3)
        self.gpool = GlobalAveragePooling2D()
        self.claasifier = Dense(num_classes , activation = tf.keras.activations.get(activation))

    def call(self , input):
        x = self.conv7(input)
        x = self.norm(x)
        x = self.pool(x)
        x = self.idbl1(x)
        x = self.idbl2(x)
        x = self.gpool(x)
        x = self.claasifier(x)

        return x

  resnet_model = Resnet(10)
  resnet_model.compile(optimizer='adam' , loss = tf.keras.losses.sparse_categorical_crossentropy , metrics=['acc'])
  ```

### 5. Week5 -> Bonus
