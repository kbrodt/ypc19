# Yandex Programming Championship 2019

## Machine Learining track

### Stage 2

#### Recommender system

Given historical user's transaction develop a recommender system for online-gallery [(description of the task in russian)](https://contest.yandex.ru/contest/12899/problems/).

#### Appoach

Take all transactions (clicks, likes and bookmarks) and create rating matrix with value

$$r_0 + s^{md - d} + esp + 1$$,

with initial rating $$r_0=50$$, smoothing parameter $$s=0.99$$,
maximum transaction date $$md$$, current date of transaction $$d$$
and small tolerance $$e=10^{-5}$$. Train 2 models from `implicit`:
- `BM25Recommender` with 600 neighbours,
- `AlternatingLeastSquares` with 512 factors and 15 iterations

and predict top 400 candidates with each model. Then blend these 3 models with weights 5.5 and 6
for `BM25Recommender` and `AlternatingLeastSquares` respectively.
The blending consist in rearranging of candidates (see `mix_solutions` in [`train.py`](train.py)).

This will reach a mean average precision at 100 (mAP@100) multiplied by 10000 around 29
(see [leaderboard](https://contest.yandex.ru/contest/12899/standings/)).

#### Usage

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=[gpus] python train.py
```
