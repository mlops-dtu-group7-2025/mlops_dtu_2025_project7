**Project description**

**Problem statement**
Flooding poses a significant threat to both rural and urban areas in South Africa, with devastating consequences for communities, infrastructure, and the economy. From 1981 to 2023, numerous floods were reported across the country, yet many events, especially in urban areas, were either underreported or not recorded with sufficient detail for further analysis or prediction work. 

The goal of the project is to classify the probability of floodings from cloudless images of the areas of interest. In the project we will try and determine the precise date a flood occurred and distinguish between time series with and without flood events.

This project is part of a competition which can be found here: https://zindi.africa/competitions/inundata-mapping-floods-in-south-africa?ref=mlcontests

**Data description**

An imbalanced dataset is provided where events are labeled with either 0 or 1, indicating whether a flood event occurred (1) or did not occur (0). Both sets of events contain randomly selected 40-60 weeks prior and post flood date, to ensure the flood event date is not always the middle index. For each location the CHIRPS precipitation data has been aggregated over a 5km radius.

For this project there has been gathered data for different locations in South Africa, each location corresponds to an event_id.

For each event_id the dataset provided consists of
:
Rainfall values stored in `train.csv` file
Satellite images obtained by Sentinel-2 for five different wavelengths, which correspond to different colours, stored in ‘composite_images.npz’.
Slope images obtained from NASA topography mission (SRTM) stored in ‘composite_images.npz’
‘test.csv’ contains the test set, consisting of 181041 time steps to label as flooded or not flooded, corresponding to 248 events. As output we need to return the probability of that event being a flooded event.

We are going to start by working with the .csv files, and then move on to images afterwards.

The composite image contains 6 bands; these bands are

Sentinel 2 B2 (Blue)
Sentinel 2 B3 (Green)
Sentinel 2 B4 (Red)
Sentinel 2 B8 (NIR)
Sentinel 2 B11 (SWIR)
slope (derived from NASA SRTM)

Frameworks and models

We intend to use the timeseriesAI (tsAI) framework to work with the time series data. It contains different models such as LSTM and ResNet, which we intend to use.

Afterwards we may also use the Pytorch Image Models (TIMM) framework to work with the images in the dataset. 


**Train with docker**

To build training docker container: 

```bash
docker build -t mlops-floods -f dockerfiles/train.dockerfile .
```
Set wandb API key: 
```bash
export WANDB_API_KEY=your_wandb_api_key
```
Run docker container with mounted data dir:
```bash
docker run --rm \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -v "$(pwd)/data:/data" \
  mlops-floods
```
