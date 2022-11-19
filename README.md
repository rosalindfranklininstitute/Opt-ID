![GitHub](https://img.shields.io/github/license/rosalindfranklininstitute/Opt-ID?kill_cache=1) [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/Opt-ID/actions/workflows/ci.yml/badge.svg?branch=v2)](https://github.com/rosalindfranklininstitute/Opt-ID/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/rosalindfranklininstitute/Opt-ID/branch/v2/graph/badge.svg?token=pZp3wgitjN)](https://codecov.io/gh/rosalindfranklininstitute/Opt-ID)

# Docker Containers

If you have podman instead of docker, simply replace the word docker with podman in the following commands.

```
# Run the container and leave it running
# Note that your current working directory on the host (where your terminal is) is being mounted into the container!
# i.e. If you run this command from a directory /dls/tmp/blah then that directory will be mounted into the container as /dls/tmp/blah.

docker run -itd --name optid -v $(pwd):$(pwd) -w $(pwd) quay.io/rosalindfranklininstitute/opt-id:v2

# Execute the tests against the container
docker exec optid python -m pytest --cov=/usr/local/Opt-ID/IDSort/src /usr/local/Opt-ID/IDSort/test/ --cov-report xml:coverage.xml

# Execute the Opt-ID main script
docker exec optid python -m IDSort.src.optid --help

# Remove the running instance of the container
docker rm optid --force
```

```
# Run the container for a single command and terminate it after
# Note the --rm flag which removes the container after the command has finished running.
# Note that your current working directory on the host (where your terminal is) is being mounted into the container!
# i.e. If you run this command from a directory /dls/tmp/blah then that directory will be mounted into the container as /dls/tmp/blah.

docker run -itd --rm --name optid -v $(pwd):$(pwd) -w $(pwd) quay.io/rosalindfranklininstitute/opt-id:v2 \
    python -m IDSort.src.optid --sort --cluster-off config.yaml
```

---

# Example: Initial Sort

Assume that we are on a command line and our current directory is /dls/tmp/blah.

We copy our initial input data into the directory such that we have the following structure:

```
/dls/tmp/blah <-- Our command line is in this directory
└── CPMU_I04
    ├── HEA.sim
    ├── HH.sim
    ├── HTD.sim
    └── config.yaml
```

Our config.yaml (below) file contains the specification for the ID that will be used to generate the .json file used for simulating the device.

Within config.yaml file paths should be relative to the directory of your terminal, /dls/tmp/blah in this example:

```
id_setup:
  type: 'Hybrid_Symmetric'
  name: 'CPMU_I04'
  periods: 113
  fullmagdims: [50.0, 30.0, 5.18]
  hemagdims: [50.0, 30.0, 3.75]
  htmagdims: [50.0, 30.0, 1.06]
  poledims: [30.0, 22.0, 2.955]
  x: [-10.0, 10.1, 2.5]
  z: [-1, 1.1, 1]
  steps: 5
  interstice: 0.35
  gap: 5.4
  endgapsym: 3.0
  terminalgapsymhyb: 3.0
  # Relative to /dls/tmp/blah this would be /dls/tmp/blah/CPMU_I04.json
  output_filename: 'CPMU_I04.json'

magnets:
  # Relative to /dls/tmp/blah this would be /dls/tmp/blah/CPMU_I04/HH.sim
  hmags: 'CPMU_I04/HH.sim'
  hemags: 'CPMU_I04/HEA.sim'
  htmags: 'CPMU_I04/HTD.sim'
  vmags: null
  vemags: null
  # Relative to /dls/tmp/blah this would be /dls/tmp/blah/CPMU_I04.mag
  output_filename: 'CPMU_I04.mag'

lookup_generator:
  random: False
  # Relative to /dls/tmp/blah this would be /dls/tmp/blah/CPMU_I04.h5
  output_filename: 'CPMU_I04.h5'

mpi_runner:
  iterations: 10
  setup: 24
  c: 1
  e: 0.0
  max_age: 10
  scale: 10.0
  verbose: 4
```

Now lets run the sort operation using the config.yaml and the .sim files!

```
docker run -itd --rm --name optid -v $(pwd):$(pwd) -w $(pwd) quay.io/rosalindfranklininstitute/opt-id:v2 \
    python -m IDSort.src.optid --sort --cluster-off CPMU_I04/config.yaml
```

Because the CPMU_I04.json, CPMU_I04.mag, and CPMU_I04.h5 don't exist yet Opt-ID will generate them and write them to the paths specified in the config.yaml.

This can take some time... and the CPMU_I04.h5 file for the lookup table can be large. For the config.yaml in this example the CPMU_I04.h5 file is 2.5 GB.

This produces the following directory structure:

```
/dls/tmp/blah
├── CPMU_I04
│   ├── HEA.sim
│   ├── HH.sim
│   ├── HTD.sim
│   └── config.yaml
├── CPMU_I04.h5
├── CPMU_I04.json
└── CPMU_I04.mag
```

Opt-ID will then start the sort optimization using randomly generated magnet orderings (genomes) to seed the optimization process.

With verbosity set to 4 in the config.yaml INFO level messages will be reported to the console output. Verbosity of 5 will show DEBUG level messages.

```
...
...earlier output omitted here...
...
13:49:24.933     INFO | MainThread | mpi_runner::process::284  | Iteration 7
13:50:26.088     INFO | MainThread | mpi_runner::process::303  | Node   0 of   1 updated estar 0.00056747
13:50:26.088     INFO | MainThread | mpi_runner::process::311  | Saving best genome 6220e6b9fdec with fitness 5.73197437E-04 age 0 mutations 8
13:50:26.091     INFO | MainThread | mpi_runner::log_genomes::195  | Node   0 of   1 has 24 genomes with fitness (min 5.73197437E-04, max 7.20109137E-04, avg 6.81956885E-04) age (min 0, max 0, avg 0.00) mutations (min 1, max 9, avg 4.88)
13:50:26.093     INFO | MainThread | mpi_runner::process::284  | Iteration 8
13:51:39.397     INFO | MainThread | mpi_runner::process::303  | Node   0 of   1 updated estar 0.00046050
13:51:39.397     INFO | MainThread | mpi_runner::process::311  | Saving best genome d9ba24eeb87c with fitness 4.65146840E-04 age 0 mutations 10
13:51:39.400     INFO | MainThread | mpi_runner::log_genomes::195  | Node   0 of   1 has 24 genomes with fitness (min 4.65146840E-04, max 5.71342831E-04, avg 5.47002401E-04) age (min 0, max 0, avg 0.00) mutations (min 1, max 10, avg 7.00)
13:51:39.401     INFO | MainThread | mpi_runner::process::284  | Iteration 9
13:52:51.071     INFO | MainThread | mpi_runner::process::303  | Node   0 of   1 updated estar 0.00033952
13:52:51.071     INFO | MainThread | mpi_runner::process::311  | Saving best genome 78b9b968df50 with fitness 3.42953623E-04 age 0 mutations 3
```

This produces the following directory structure, where the best genome in the population at each iteration has been written into a ./genomes/ directory:

Note that when you list the genomes on the file system the order may be deceptive!

The file names contain the fitness of the genome in scientific notation which can cause them to be listed in the wrong order if sorted by name rather than date created.

```
/dls/tmp/blah
├── CPMU_I04
│   ├── HEA.sim
│   ├── HH.sim
│   ├── HTD.sim
│   └── config.yaml
├── CPMU_I04.h5
├── CPMU_I04.json
├── CPMU_I04.mag
└── genomes
    ├── 1.09526318e-03_000_b601507881d3.genome
    ├── 1.62618654e-03_000_cf3c2196d9b0.genome
    ├── 2.81129920e-03_000_9d25302a224a.genome
    ├── 3.42953623e-04_000_78b9b968df50.genome <-- Best Genome from the final iteration with a fitness of 0.0003429...
    ├── 3.63909109e-03_000_f88d19a77f32.genome
    ├── 4.65146840e-04_000_d9ba24eeb87c.genome
    ├── 5.58823735e-03_000_65ebfca5f669.genome
    ├── 5.73197437e-04_000_6220e6b9fdec.genome
    ├── 7.25442782e-03_000_bceba003514f.genome
    ├── 7.44059451e-04_000_5a67562fa0c7.genome
    └── 8.80150018e-04_000_59c569922c26.genome
```

Now lets process that best genome into a .h5 file so we can visualize it in DAWN:

```
docker run -itd --rm --name optid -v $(pwd):$(pwd) -w $(pwd) quay.io/rosalindfranklininstitute/opt-id:v2 \
    python -m IDSort.src.process_genome \
           -m CPMU_I04.mag -t CPMU_I04.h5 -i CPMU_I04.json \
           -o processed -a genomes/3.42953623e-04_000_78b9b968df50.genome
```

This produces the following directory structure:

```
/dls/tmp/blah
├── CPMU_I04
│   ├── HEA.sim
│   ├── HH.sim
│   ├── HTD.sim
│   └── config.yaml
├── CPMU_I04.h5
├── CPMU_I04.json
├── CPMU_I04.mag
├── genomes
│   ├── 1.09526318e-03_000_b601507881d3.genome
│   ├── 1.62618654e-03_000_cf3c2196d9b0.genome
│   ├── 2.81129920e-03_000_9d25302a224a.genome
│   ├── 3.42953623e-04_000_78b9b968df50.genome <-- Best Genome from the final iteration with a fitness of 0.0003429...
│   ├── 3.63909109e-03_000_f88d19a77f32.genome
│   ├── 4.65146840e-04_000_d9ba24eeb87c.genome
│   ├── 5.58823735e-03_000_65ebfca5f669.genome
│   ├── 5.73197437e-04_000_6220e6b9fdec.genome
│   ├── 7.25442782e-03_000_bceba003514f.genome
│   ├── 7.44059451e-04_000_5a67562fa0c7.genome
│   └── 8.80150018e-04_000_59c569922c26.genome
└── processed
    └── 3.42953623e-04_000_78b9b968df50.genome.h5 <- Processed Genome ready to view in DAWN
```

From this example you should be able to see how Opt-ID v2 operating within a container is interacting with the file system.

All other Opt-ID v2 scripts and features will work with the file system in the same way.