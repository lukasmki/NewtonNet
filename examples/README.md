# NewtonNet Examples

Run the driver (`model_train.py`) in the example folders.

```bash
cd examples/train-cpu
python3 model_train.py
```

Example filepaths are relative to the directory the driver is contained in.

An example filestructure would be the following, with `model_train.py` being run from the `project` dir.

```bash
project/
 ├── model_train.py
 ├── output/
 |    ├── training_1/
 |    └── training_2/
 └── dataset/
      ├── data1.npz
      └── data2.npz
```
