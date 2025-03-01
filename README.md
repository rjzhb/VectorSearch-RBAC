## Prerequisites

- PostgreSQL
- psycopg2

### Clone the Repository

Clone this repository to your local machine.

### Install PostgreSQL 16 or higher and Development Tools
Ensure PostgreSQL and necessary development tools are installed

```sh
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all build-essential
```
```shell
sudo apt install libpq-dev
```

### Install pgvector
Ensure Pgvector is installed

### Setup Python Environment
Create a virtual environment and install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setup embedding model

```sh
python -m spacy download en_core_web_md
```

### Download Dataset

Download the dataset to {project directory}/dataset/:

- [Cohere Wikipedia 22-12 Dataset](https://huggingface.co/datasets/Cohere/wikipedia-22-12)
```shell
cd rbac-bench
git lfs install
git clone https://huggingface.co/datasets/Cohere/wikipedia-22-12
```


### Configure database config file

Edit 'config.json' in project root directory.

```json
{
    "dbname": "rbacdatabase",
    "user": "database_user",
    "password": "database_password", 
    "host": "localhost",
    "port": "5432"
}
```
### Run postgresql

```shell
 sudo service postgresql start
```

### Prepare Data
```sh
cd benchmark
# adjust load_number to controll data scalability
python3 common_prepare_pipeline.py

go to controller directory
# prepare main tables(user/role/permission) index
python3 initialize_main_tables.py
```

### Generate Permission
```sh
go to services/rbac_generator diretory

# Taking treebased as an example
python3 store_tree_based_rbac_generate_data.py
```
### Initilize partition and prepare for queries
```shell
go to basic_benchmark
# initialize role partition
python3 initialize_role_partition_tables.py

# (optional) initialize user partition
python3 initialize_combination_role_partition_tables.py

# generate queries
python3 generate_queries.py --num_queries 1000 --topk 10 --num_threads 4
```

### Initilize honeybee partition
```sh
# initilize HONEYBEE
go to controller/dynamic_partition/hnsw
# if needed, delete parameter_hnsw.json from hnsw directory to regenerate parameters
python3 honeybee_dynamic_partition.py --storage 2.0 --recall 0.95
```

Run(HNSW index)
```sh
go to basic_benchmark directory

# example:
python test_all.py --algorithm RLS --efs 20 30 40 50
python test_all.py --algorithm ROLE --efs 20 30 40 50
python test_all.py --algorithm USER --efs 20 30 40 50
python test_all.py --algorithm HONEYBEE --efs 20 30 40 50
```


Run(ACORN index)
```sh
go to acorn_benchmark directory

#modify efs value from main.cpp
build C++ project and run
```
