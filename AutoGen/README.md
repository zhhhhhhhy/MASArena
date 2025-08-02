## quick start

1、download

```cmd
git clone -b relative-works-official-codebase https://github.com/LINs-lab/MASArena.git
```

2 、Switch to the Autogen directory

```cmd
cd MASArena
cd AutoGen
```

3、Configure the environment

```cmd
uv sync
```

4、Add .env file to add APIs

Format refer to .env.example

4、start 

```cmd
python main.py --benchmark [datasetname] --agent-system autogen --limit 1
```

example：

```cmd
python main.py --benchmark bbh --agent-system autogen --limit 1
```





#### Experimental Results

| Version | math   | AIME   | DROP   | MMLU_pro | BBH    | Humaneval |
| ------- | ------ | ------ | ------ | -------- | ------ | --------- |
| Origin  | 49.00% | 10.00% | 87.00% | 64.00%   | 77.00% | 77.00%    |
| Arena   | 52.00% | 16.70% | 91.00% | 59.00%   | 66.00% | 78.00%    |

