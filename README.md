# SWIMMER TRACKING

The aim of this code is to provide two different methods of swimmer tracking. The inputs video are filmed with a fixed camera.

- The first one used computer vision techniques to catch the arms motion frequency of each swimmer.
- The second is a Deep Learning model. With a simple CNN architecture, we find the (discrete) position of each swimmer, at each time of the video.

<td><img width="400px" src="output/gif/first_method_detection.gif"></td>

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
    LINES_TO_PLOT = [1]
```

**To test the second method**, run
```bash
python ????????
```
## Contributors
Victoria Brami, Maxime Brisinger, Rémi Carmigniani, and Théo Vincent 



## License
[ENPC](https://www.ecoledesponts.fr/)
