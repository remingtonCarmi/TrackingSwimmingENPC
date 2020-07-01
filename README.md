# SWIMMER TRACKING

The aim of this code is to provide two different methods of swimmer tracking. The input videos are filmed with a fixed camera.

- The first one used computer vision techniques to catch the arms motion frequency of each swimmer.

<DIV ALIGN="CENTER">
<td><img width="600px" src="data/4_model_output/gif/first_method_detection.gif"></td>
</DIV>

- The second is a Deep Learning model. With a simple CNN architecture, we find the (discrete) position of each swimmer, at each time of the video. 
To achieve our goal, we built an application to label videos, and then created our own dataset.

<DIV ALIGN="CENTER">
<td><img width="600px" src="data/4_model_output/gif/second_method_detection_10_classes.gif"></td> <br>
<td><img width="600px" src="data/4_model_output/gif/second_method_detection_30_classes.gif"></td>
</DIV>




## Prerequisites & Installation:

need Python version 3.7

To install all the required libraries run the command 
				

```bash
python requirements.py
```

In *"data\videos\"*, please put all the following videos :
- *vid0.mp4*
- *vid1.mp4*
- *100NL_FAF.mov.mp4*
- *2004N_FHA.mov.mp4*

These videos are provided by the *Fédération Française de Natation*. Feel free to ask us how to download them.

**Note** : on Mac OS, some functionnalities might not work. Please tell us when you point some of them out. We'll fix them asap.
## Usage

**To test the first method**, run
```bash
python main_rough_detection.py
```
Before, feel free to modify the line 22 if you want the graph of more swimmers
```python
    # lanes we want to plot the swim frequency
    LANES_TO_PLOT = [1]
```

**To test the second method**, run
```bash
python main_classification.py
```

**To label a video**, run
```bash
python create_data_set.py
```
## Contributors
Victoria Brami, Maxime Brisinger, Rémi Carmigniani, and Théo Vincent 



## License
[ENPC](https://www.ecoledesponts.fr/)
