# Counting-Sheep
Counting Sheep without Sleep. This project utilizing YOLO and several image processing techniques to track, count and segment sheep.

## How to run this repository
I recommend creating an anaconda environment:
```
conda create --name [environment-name] python=3.9
```

Then, install Python requirements:
```
pip install -r requirements.txt
```
Finally, to reproduce the results, from the `[environment-name]` project root, run:
```
python counting_sheeps.py
```

**Important**: The code within the `drawing_bounds.py` file handles the crucial task of delineating and extracting the detection area. It will need adjustments to accommodate various videos and camera angles, or possibly even be disabled. Additionally, employing a stable camera in the video setup, ideally capturing all sheep from a high angle  (even though I lack this type of video), will significantly improve accuracy.

## Demo
Here are some demos showcasing the results obtained from videos sourced from **YouTube**.

https://github.com/khoi03/Counting-Sheeps/assets/80579165/5ddacd24-a270-4521-8807-8d1d61b0edb7

https://github.com/khoi03/Counting-Sheeps/assets/80579165/efc5e2b9-36cb-4d99-a7c6-f08da096e6b5
