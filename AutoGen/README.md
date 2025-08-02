## quick start

1、download

```cmd
git clone -b relative-works-official-codebase https://github.com/LINs-lab/MASArena.git
```

2、Configure the environment

```cmd
cd MASArena
uv sync
```

3、Add .env file to add APIs

Format refer to .env.example

4、start 

```cmd
cd AutoGen
python main.py --benchmark [datasetname]  --limit [Number of questions]
```

example：

```cmd
python main.py --benchmark bbh --limit 1
```

Note: There is a package conflict between Autogen and ChatDev, the version of Pillow used by ChatDev is 10.3.0, and Autogen requires a higher version, if ChatDev cannot be used due to conflicts, please delete the package of autogen and change the Pillow of requirements to Pillow == 10.3.0

