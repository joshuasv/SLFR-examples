# Sign Language Fingerspelling Recognition (SLFR) Examples

## Virtual environment

Miniconda was used to manage the virtual environment, and it is suggested to do 
so. To get the environment up and running, execute:
```bash
conda env create -f environment.yml
```

## Dataset

A copy of the dataset from the American Sign Language Fingerspelling Recognition
Kaggle competition is located in `/home/temporal2/jsoutelo/datasets/GSLFR`. It 
is suggested to create a symlink to it inside the `./data` folder. Run:

```bash
ln -s  /home/temporal2/jsoutelo/datasets/GSLFR absolute/path/to/data
```

### Data splits

The data splits are located inside the `./data_gen` directory. Just as training
(or inference), they have a configuration file associated to each of them, they
can be found in `./data_gen/config`. To generate a new split run:

```bash
python -m data_gen.generate_splits --config ./data_gen/config/config_file.yaml
```

As a result, a folder containing the splits will be created. It will have the 
same name as the configuration file used to generate them. I woudl recommend
to start by generating the `baseline.yaml` split.

### Raw data

In case that you want to use the raw data from the competition directly, it is 
provided two scripts inside `./raw_data_scripts` folder. They handle the 
extraction logic of the raw `.parquet` files. This is not recommended.

## Fine-tuning and Inference

To switch between the two modes it is just enough to edit the `phase` field in
the config file to run either `train` or `test`. Once configured, simply execute
the following:

```bash
python main.py --config ./path/to/config/file.yaml
```

Two examples are provided in the `./config` folder for reference.

## Notes

### Problems with `torch_edit_distance`

If some kind of problem airses with the package `torch_edit_distance` please
reinstall. A compiled version for the `baiona` server can be found in
`/home/temporal2/jsoutelo/builds/pytorch-edit-distance-bai`. With a virtual 
environment activated, run:

```bash
cd /home/temporal2/jsoutelo/builds/pytorch-edit-distance-bai
python setup.py install
```

If the error persists feel free to reach out to me.