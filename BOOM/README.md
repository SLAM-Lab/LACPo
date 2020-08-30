-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
src:
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Source for the BOOM core is written in Chisel hardware construction language. Chipyard project template is used to generate the RTL for simulation and synthesis. The src directory points to the chipyard repo and below are the steps to generate the medium config BOOM core (from https://docs.boom-core.org/en/latest/):

# Download the template and setup environment
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard
./scripts/init-submodules-no-riscv-tools.sh

# build the toolchain
./scripts/build-toolchains.sh riscv-tools

# add RISCV to env, update PATH and LD_LIBRARY_PATH env vars
# note: env.sh generated by build-toolchains.sh
source env.sh

#verilator simulation
cd sims/verilator
make CONFIG=MediumBoomConfig

#VCS simulation
cd sims/vcs
make CONFIG=MediumBoomConfig

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
synth_PD
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TCLs from KIT - Netlist generated at synth_PD/BoomTile.v

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
simulation
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The commands for the VCS simulation of RTL & netlist are as follows:

#Object file creation - RTL  : Replace all <x> with corresponding paths
vcs -LDFLAGS -no-pie -full64 -notice -line -CC -I<VCS_HOME>/include -CC -I src/chipyard/riscv-tools-install/include -CC -std=c++11 -CC -Wl,-rpath,-I src/chipyard/riscv-tools-install/lib  src/chipyard/riscv-tools-install/lib/libfesvr.so  +lint=all,noVCDE,noONGS,noUI -error=PCWM-L -timescale=1ns/10ps -quiet -q +rad +v2k +vcs+lic+wait +vc+list -f  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/sim_files.common.f -sverilog +incdir+ src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig +define+CLOCK_PERIOD=3.0  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.top.v  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.harness.v  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.top.mems.v  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.harness.mems.v +define+PRINTF_COND=TestDriver.printf_cond +libext+.v -o  src/chipyard/sims/vcs/simv-MediumBoomRTL_clkconstraints_initialize_with_zero_vcd -debug_pp +vcs+initreg+random +vcs+dumpvars+<vcd_path>/temp_tile.init0.rtl.vcd
#Object file creation - PD  : Replace all <x> with corresponding paths
vcs -LDFLAGS -no-pie -full64 -notice -line -CC -I<VCS_HOME>/include -CC -I src/chipyard/riscv-tools-install/include -CC -std=c++11 -CC -Wl,-rpath,-I src/chipyard/riscv-tools-install/lib  src/chipyard/riscv-tools-install/lib/libfesvr.so  +lint=all,noVCDE,noONGS,noUI -error=PCWM-L -timescale=1ns/10ps -quiet -q +rad +v2k +vcs+lic+wait +vc+list -f  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/sim_files.common.f -sverilog +incdir+ src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig +define+CLOCK_PERIOD=3.0  <NangateOpenCellLibrary.syn.vcs.v> synth_PD/BoomTile.v  <example.TestHarness.MediumBoomConfig.top.edited2.v>  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.harness.v  src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.harness.mems.v +define+PRINTF_COND=TestDriver.printf_cond +libext+.v -o  src/chipyard/sims/vcs/simv-MediumBoomRTL_clkconstraints_initialize_with_zero_vcd_pd -debug_pp +vcs+initreg+random +vcs+dumpvars+<vcd_path>/temp_tile.init0.pd.vcd
Where : 
  VCS_HOME - Pointer to VCS installation directory - Version used - M-2017.03-SP2
  NangateOpenCellLibrary.syn.vcs.v - Standard cell definitions - Nangate library
  BOOMTILE_NETLIST - PD netlist of the BOOM Tile
  example.TestHarness.MediumBoomConfig.top.edited2.v - Edited version of src/chipyard/sims/vcs/generated-src/example.TestHarness.MediumBoomConfig/example.TestHarness.MediumBoomConfig.top.v to remove duplicate module definitions (sample src/chipyard/sims/vcs/example.TestHarness.MediumBoomConfig.top.edited2.v)
#Simulation dump generation  - RTL - sample test - median. It generates <vcd_path>/temp_tile.init0.rtl.vcd
src/chipyard/sims/vcs/simv-MediumBoomRTL_clkconstraints_initialize_with_zero_vcd +permissive +vcs+initreg+0 +max-cycles=10000000 +verbose  +permissive-off chipyard/riscv-tools-install/riscv64-unknown-elf/share/riscv-tests/benchmarks/median.riscv > median.rtl.init_with_zero.log
#Simulation dump generation  - PD - sample test - median. It generates <vcd_path>/temp_tile.init0.rtl.vcd
src/chipyard/sims/vcs/simv-MediumBoomRTL_clkconstraints_initialize_with_zero_vcd_pd +permissive +vcs+initreg+0 +max-cycles=10000000 +verbose  +permissive-off chipyard/riscv-tools-install/riscv64-unknown-elf/share/riscv-tests/benchmarks/median.riscv > median.pd.init_with_zero.log


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Feature engineering
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simulation to CSV: 
   - The script for post-processing the VCD for feature generation is ../common/vcd2csv/vcd2csv.py. Steps to run are described in README in the same directory.
   - ../common/vcd2csv_post has the helper script to generate launch commands for running vcd2csv and post processing it.
   - Input signal list can be generated from the verdi signal export list using the script common/vcd2csv_post/post_process_the_signal_list.sh
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
power_anal
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The vcd_path file should be edited beforehand to list the vcds that need to be used for power analysis. To compute average power, source pt_avg_pwr.tcl in pt_shell. For time based power, use pt_timebased_pwr.tcl.
Other scripts:
-------------
Average power:
-------------
get_hier_power.pl - Get the hierarchy number for parsing the hier power.
  Input  : List of hierarchy to extract power - one in each line
  Input  : avg rpt with the line number
  Output : Instance name,Line number,hierarchy level 
  Usage  : perl get_hier_power.pl hier_to_get_frontend_stripped hier_numbers frontend_hier_numbers

parse_hier_power.pl - parses the hierarchical power report and generates csvs.
  Input  : lines_to_get - comma separated file containing line number corresponding to the module and its level in the hierarchy (BoomTile is in level-0)
  Input  : *.power_hier.avg.rpt
  Output : <Output_file>.csv
  Usage  : perl parse_hier_power.pl lines_to_get median.power_hier.avg.rpt median.power_hier.avg.csv

merge_rpts.pl - Generate merged representation of all the average power CSVs .
  Input  : list_of_csvs - list of all csvs that needs to be merged
  Output : <Output>.merged.csv
  Usage  : perl merge_rpts.pl list_of_csvs top_csvs.merged.csv

Time based power:
----------------
parse_out.pl - Generate parsed, post-processed form of out file from PTPX
  Input  : out file from PTPX
  Output : <output_file>.out.parsed
  Usage  : perl parse_out.pl ../median.out parsed_csvs/median.out.parsed

get_shell_script_for_csv_generation.pl - Generate the shell script that can be used for generating the csvs for the blocks
  Input  : hierarchies_number.rpt - Hierarchy numbers for the required modules (can be generated from one of the *parsed output files)
  Input  : list_of_parsed_out - List of parsed files 
  Output : run_me.gen_csv.sh
  Usage  : perl get_shell_script_for_csv_generation.pl hierarchies_number.rpt list_of_parsed_out run_me.all_blocks.sh

run.generic.sh  - Base shell file used for generating csvs for the blocks - The run_me.gen_csv.sh creates a file calling run.generic.sh for different blocks and parsed files.

parsedout_to_csv.pl -  Called within run.generic.sh
  Input  : *parsed_out file
  Output : *parsed.csv

 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
power_modeling:   
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Step 1 : 
  signal_power_mapping - The power csvs generated in power_anal and feature csvs from feature_engineering are merged based on the cycle key. The resulting csvs are used for the power modeling.
  data_prep.just_merge.py - Merge power and signal csv
  data_prep.90_10_split.py - Merge and do a 90_10 split (no of folds is an argument)
  data_prep.benchmark_split.py - Merge and do benchmark based split
    Usage :
     python data_prep.just_merge.py --signals <extracted_features>.csv --power <parsed_power>.csv --output <output>
     python data_prep.90_10_split.py --signals  <extracted_features>.csv --power <parsed_power>.csv --output <output> --folds <no_of_folds_to_generate>
     python data_prep.benchmark_split.py --signals <extracted_features>.csv --power <parsed_power>.csv --output <output> --folds 1 --benchmark_for_test <benchmark_name>

Step 2 : 
  Power model generation and validation - Use regression_models.py
  Power model invocation                - Use regression_models.invoc.py


power_modeling/power_models has pre-trained models that can be retrieved and used with regression_models.invoc.py script.