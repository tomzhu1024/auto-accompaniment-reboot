# auto-accompaniment-reboot

An automatic accompaniment system for vocal performance (Python rewritten version).

Rewritten from [MichaelZhangty/Auto-Acco-2020](https://github.com/MichaelZhangty/Auto-Acco-2020). Enhanced execution performance and improved project structure.

## Structure Description

| Name | Description |
| :--- | :--- |
| `/bin` | Binary executables |
| `/resources` | Static resources |
| `/utils` | Supporting libraries |
| `/output` | Execution results |
| `/playground` | Experimental stuffs |
| `global_config.py` | Global configuration file |
| `score_following.py` | Entry of score following module |
| `accompaniment.py` | Entry of accompaniment module |
| `plotter.py` | Entry of plotting module |

## Configuration Description

See comments in `global_config.py`.

## How to use

Start accompaniment module:

```shell
python accompaniment.py
```

Then, start score following module:

```shell
python score_following.py
```

*Note that* two modules need to run in parallel. Run them in two different shells if you use shell.

## To-Do List

- [ ] Refine visualization tools
- [ ] Tune parameters
- [ ] Introducing simple ML techniques
- [ ] Merge multiple entry points
- [ ] Replace UDP IPC pipe
- [ ] Migrate to C++
