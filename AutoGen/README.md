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


