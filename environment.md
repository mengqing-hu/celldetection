
#### Use srun to enter compute node:

```
srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash -l

```





#### Check workspace on Capella:

```
ws_list -t
ws_list -s
```






#### Module management on Capella:

```
module spider Python/3.13.5
module load release/2026  GCCcore/14.3.0
module load Python/3.13.5
module list
module --force purge
module unload Python/3.13.5
module avail Python/3.13.5
module show Python/3.13.5

```



#### Create virtual environment and install dependencies on Capella:

```
python -m venv .venv
source .venv/bin/activate
rm -rf venv
which python
deactivate
pip install -r requirements.txt
pip freeze > requirements.txt

```


#### Jupyter kernel management on Capella:

```
pip install ipykernel
pip install --upgrade pip
python -m ipykernel install --user --name cpn-kernel --display-name="cpn kernel"



jupyter kernelspec list
jupyter kernelspec uninstall cpn-kernel

```





