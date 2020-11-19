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

* Week1 -> Funcitional API 
  1. Create a model Using functional API( Notebook ->  [1st_notebook.ipynb](https://github.com/ANKITPODDER2000/Tensorflow-Advance/blob/main/Custom_Models_Layers_and_Loss_Functions_with_TensorFlow/Week1/1st_notebook.ipynb) )
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
* Week2 -> Custom Loss Function
* Week3 -> Custom Layers
* Week4 -> Custom Models
* Week5 -> Bonus
