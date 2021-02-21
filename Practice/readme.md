# Practice of Adv Tensorflow

* Regression (Data Source : Youtube Channel)
  #### Model Summary :
  
  * Linear Model( f(x) = W x X + B )
  <p align="center">
    <img src="https://user-images.githubusercontent.com/50513363/99941181-95b45d00-2d93-11eb-9de6-13b3a18832fb.gif"/>
  </p>
  
  * Polynomial model (deg - 2)
  <p align="center">
    <img src="./Regression/animation_deg_2.gif"/>
  </p>
  
  * Polynomial model (deg - 4)
  <p align="center">
    <img src="./Regression/animation_deg_4.gif"/>
  </p>
  
  * Polynomial model (deg - 6)
  <p align="center">
    <img src="./Regression/animation_deg_6.gif"/>
  </p>
* [Cat & Dog Classification](https://ankitpodder2000.github.io/cat_dog_detect/)

  * *Prediction :*
  <p align="center">
    <img src="./Cat_Dog/animation_mod.gif"/>
  </p>
  
  * *Model Accurecy and loss on training and testing data*
  <p align="center">
    <img src="./Cat_Dog/animation_loss_mod.gif"/>
  </p>
  
  
* [handwriting_detect](https://ankitpodder2000.github.io/handwriting_detect/)
  
  ***
  
  ##### Model details

  ```python

  def get_base_model():
      #input = tf.keras.layers.Input(shape = (28,28,1))
      #x = tf.keras.layers.Dense(64 , activation=tf.nn.relu)(input)
      #x = tf.keras.layers.MaxPool2D((2,2))(x)
      #x = tf.keras.layers.Dense(128 , activation=tf.nn.relu)(x)
      #x = tf.keras.layers.MaxPool2D((2,2))(x)

      #x = tf.keras.layers.Flatten()(x)
      #x = tf.keras.layers.Dense(256 , activation=tf.nn.relu)(x)
      #x = tf.keras.layers.Dense(26 , activation=tf.nn.sigmoid)(x)

      #model = tf.keras.Model(inputs = input ,outputs = x)
      #model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) ,
                    #loss = tf.keras.losses.categorical_crossentropy , metrics = ['acc'])
      #return model

      cls = Sequential()
      cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
      cls.add(MaxPooling2D(pool_size=(2, 2)))
      cls.add(Dropout(0.3))
      cls.add(Flatten())
      cls.add(Dense(128, activation='relu'))
      cls.add(Dense(26, activation='softmax'))

      cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      return cls
  ```

  ##### Conver keras model into tensorflow.js model

  ```python
  !pip install tensorflowjs
  import tensorflowjs as tfjs
  tfjs.converters.save_keras_model(model1 , 'model')
  ```


  ##### UI of the web APP
  ---
  <p align="center">
    <img width="600px" src="https://user-images.githubusercontent.com/50513363/100707335-873bf600-33d0-11eb-85c4-4009c5ffbf93.png" />
  </p>
  
  <p align="center">
   <img width="600px" src="https://user-images.githubusercontent.com/50513363/100707529-d7b35380-33d0-11eb-88d9-6515598f7cf2.png" />
  </p>
  
  * ***Realtime Object detection***

  --------------------
  ##### Important Block of code
  ```
    model_link = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector   = hub.load(model_link).signatures['default']
  ```
  ##### Block of code for creating video
  ```
    def real_time_detect(File_path , print_details = True):
      for i in os.listdir("images/"):
          os.remove("images/"+i)
      cap = cv2.VideoCapture(File_path)
      print("Process started ...")
      count = 0
      frame = None
      while cap.isOpened():
          ret, frame = cap.read()
          if ret:
              result = detector(tf.image.convert_image_dtype(frame , tf.float32)[tf.newaxis , ])
              result = {k:v.numpy() for k , v in result.items()}
              res = draw_boxes(
                  frame ,
                  result['detection_boxes'] ,
                  result['detection_class_entities'] ,
                  result['detection_scores'] ,
                  max_boxes=20
              )
              im_name = "try"+str(count)+".png"
              cv2.imwrite("images/"+im_name , res)
              if count % 25 == 0:
                  print("%d images are Created ..."%count)
              count += 1
          else:
              break
      cap.release()
      print("Process done ......")
      print("Creating Video .........")

      img = []
      for i in os.listdir("images/"):
          img.append(cv2.imread("images/"+i))
      print("Total no of images : ",len(img))
      # Writes the the output image sequences in a video file


      imageio.mimsave('./animation_last_25fps.mp4' , img , fps = 25)
      print("Video creartion done ......")
  ```
  <p align="center">
    <img src="Real Time Object Detection/test.gif" />
  </p>
