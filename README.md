# 2627_LTXLDL_Airbnb_Nhom10
Project Lập trình xử lý dữ liệu - Inside Airbnb - Nhóm 10 - UET

# Installation
You will need to install pip first through your terminal. After that, change your directory in terminal to the root of the project.

For example, mine is like this:

```shell
cd C:/Documents/<Project_name>
```

After that just run:

```shell
pip install -r requirements.txt
```

It will overwrite all of your existing packages that are also in the requirements file

# How to run
Just paste the below code in your terminal and run after you're done with the installation

```shell
python run_pipeline.py --config configs/<city_name>.yaml
```

Replace the <city_name> with your respective city

# API Key's variable's name

It should, hopefully, be this

```shell
GEMINI_API_KEY
```

Save your api key in .env with the name above

# Where to change city, snapshots

You can find it in the configs folder, there should be a file name florence.yaml
