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

### 3. Week3 -> Custom Layers

### 4. Week4 -> Custom Models

### 5. Week5 -> Bonus
