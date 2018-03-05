# image_caption

Image caption is a task to generate a caption to describe given image. The dataset is MS COCO， which is a large-scale object detection, segmentation, and captioning dataset. 

To train the model, we take the following actions:
- Use VGG16 to extract image features from the last fully connected layer. 
- Feed the images features as the initial hidden state of LSTM
- Combine the image features and input sequences, train the LSTM to predict the next word of current image caption. We use caption to self-predicted the next word.
- Use a fully connected layer to predict the probabilily of next word over the whole dictionary.
