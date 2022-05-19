# neuralpick
Phase picking and quality checking library for use with seismic data. Experimental Python wrapper tool that allow users to train and use computer vision models for characterizing their own seismic waveform data sets


WIP, currently halted due to work conflicts of interest.


Created by Jorge A Garcia, 2020


Train your own model, which can be used to pick seismic phases in new seismograms.


Quick visual demonstration of the basic scanning function after obtaining your trained model. The gray are the sliding input windows being fed into the model, with the red being the predicted pick. Notice the consistent picks when the arrival peak is enclosed (can be trained similarly on the onset).
![Pick scan](https://github.com/JorgeAGR/neuralpick/blob/master/src/movie_arrival_pick.gif)
