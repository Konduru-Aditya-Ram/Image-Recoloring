# Image-Recoloring

To build a recoloring model we used traditional method that is to convert an image into LAB colour space and train the model on L channel(lightness channel typically a black and white image ) as input and AB Channels as outputs 
Model Architecture:
    Encoder-Decoder:This network consist of an encoder that compresses the input image into a lower-dimensional representation and a decoder that reconstructs the colorized image from this representation.And we used MSE(Mean square error)as loss function.  

The model can be readily located within the 'models' file, while the code required to utilize the model is available in the 'Encoder-Decoder' file.



Resources used for learning concepts:
        Fundamentals of image processing:https://realpython.com/lessons/how-computers-see-images/#:~:text=They%20operate%20on%20binary%20values,are%20all%20called%20color%20channels.
        OpenCv :https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
        ML:https://youtu.be/GwIo3gDZCVQ?si=u0wlpo01sjPEj08l
        CNN:https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
Sources used for collecting images:        
        https://www.pexels.com/
        https://unsplash.com/
