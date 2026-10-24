# run.sh

Build and run the HIP unit tests against the spvcpu CPU backend, one test at a time.

## Usage

```bash
./run.sh                  # all enabled unit tests
./run.sh -c <pattern>     # only tests matching <pattern> (grep, BRE)
./run.sh --clean          # delete the build dir, then reconfigure and run
```

Combine flags: `./run.sh --clean -c <pattern>`.

`-c` is a grep filter over the target path; use `\|` for "or":

```bash
./run.sh -c 'unit/compiler/'
./run.sh -c 'unit/kernel/'
./run.sh -c 'unit/deviceLib/'
./run.sh -c 'unit/device/'
./run.sh -c 'unit/context/'
./run.sh -c 'unit/device_memory/'
```

## Results

```text
results/<area>/<name>/build.log   # compile output
results/<area>/<name>/run.log     # run output
results/success.txt               # passed: area/name per line
results/todo.csv                  # failed: area/name,build or area/name,run
results/000_records/<stamp>/      # per-run copy of success.txt / todo.csv
```
