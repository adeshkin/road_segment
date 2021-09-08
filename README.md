# Road Segment Identification Hackathon: 7th place solution

![header](images/header.png)

- 🛆[Road Segment Identification Hackathon](https://zindi.africa/hackathons/road-segment-identification-challenge)


## Dataset

Download
[dataset](https://zindi.africa/hackathons/road-segment-identification-challenge/data)

## Environment:
```bash
git clone https://github.com/adeshkin/road_segment.git 
cd road_segment
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```
## Training

```bash
python main.py default
```

## Submit
- `./submissions_folds/ensemble_10x.csv`:
```bash
python main.py ensemble_10x
```
- `./submissions_folds/ensemble_5x.csv`:
```bash
python main.py ensemble_5x
```







