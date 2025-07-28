# Large Language Diffusion Model

This repository is an implementation of Large Language Diffusion Model (https://arxiv.org/abs/2502.09992)

### Install
```bash
pip install requirements.txt
```

### Train

```bash
python train.py [--arg value ...]
```

| Argument          | Type  | Default         | Description                                                         |
| ----------------- | ----- | --------------- | ------------------------------------------------------------------- |
| `--config`        | str   | required        | Path to the train config file                                       |
| `--wandb `        | bool  | `False`         | Use Weights&Biases for experiment tracking                          |


### Inference

```bash
python reverse_process.py [--arg value ...]
```


| Argument           | Type  | Default         | Description                                                         |
| ------------------ | ----- | --------------- | ------------------------------------------------------------------- |
| `--config`         | str   | required        | Path to the train config file                                       |
| `--checkpoint`     | str   | required        | Path to the trained model checkpoint                                |
| `--prompt`         | str   | `""`            | Prompt for text generation                                          |
| `--num_steps`      | int   | `50`            | Steps of the reverse process                                        |
| `--do_sampling`    | bool  | `False`         | Distribution sampling during token selection                        |
| `--low_conf_remask`| bool  | `False`         | Use low confidence remasking strategy                               |
