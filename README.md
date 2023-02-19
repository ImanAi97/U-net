# Image Segmentation using U-net

In this project we are goint to tackle <b>Carvana Image Masking Challenge</b> from kaggle.<br>

Dataset can be found here https://www.kaggle.com/competitions/carvana-image-masking-challenge/data

# Database Samples
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/0cdf5b5d0ce1_02.jpg)
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/mask0cdf5b5d0ce1_02.jpg)
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/0d53224da2b7_11.jpg)
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/mask0d53224da2b7_11.jpg)

# Training the model
We can train the model with free GPU of google colab using <b>train.py</b><br>

## Performance evaluation
We used <b>dice score</b> and <b>pixel accuracy</b> as evaluation metrics for our binary segmentation task.


Finally we achieve near 99 percent dice accuracy with just 7 epochs on the validation set.
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/Screenshot%20from%202023-02-19%2018-40-21.png)

## Testing 

Using trained model, we can now inference the mask image from the normal image by using <b>inference.py</b>

### Testing results
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/0.png)
![image](https://github.com/ImanGoudarzvand/U-net/blob/master/unet%20pics/10.png)







