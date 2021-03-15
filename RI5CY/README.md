This directory holds the instructions and scripts used for the power modeling of the RI5CY core that is part of the PULP platform.

RTL source:
----------
Source of the core can be checked out from the github repository - tag pulpissimo-v2.0.0
Repo : https://github.com/openhwgroup/cv32e40p.git
Website : [https://github.com/openhwgroup/cv32e40p/tree/pulpissimo-v2.0.0]

Simualtion:
----------
RTL/Gate-Level simulation of the core is done using the pulpino platform testbenches and benchmark were
chosen from the pulpino test suites.
Repo : https://github.com/pulp-platform/pulpino.git

The benchmarks were compiled to generate the executable as well the disassembled code. Script 'write_stim_file_upd.pl' can be used to convert the disassembled code to a hexadecimal stimulus for the simualtion. If the compressed code generation is enabled, it handles the merging required to be compliant with the endianness of the RI5CY core.
The memory loading file - l2_stim.slm and tcdm_bank0.slm - are updated with appropriate stimulation files  in tb/slm_files directory. Simulation dump time can be controlled by tweaking the delay before $finish in tb.sv. Execution start address (ROM_START_ADDR) in peripherals.sv file is modified to point to the main function in the compiled benchmark. Simulation can be run using the simulate.sh.
Reset values of the r1 (return address) and r2 (stack pointer) is edited to FFFF and 7FF0 respectively to avoid overwriting of the program code in riscv_register_file.sv.

Synthesis: syn.tcl can be used for synthesizing the core. 

Power analysis: Time based power analysis can be done using ptpx.tcl.

Feature extraction:
------------------
The feature extraction from the VCD can be done using the Cadence Simvision. Sample svcf files to load the required signals corresponding to the features are added to the svcfs directory; these signals can then be exported to csvs. The csv file can then be postprocessed to make the splits and compute hamming and appended with the power trace from the power analysis. 

Power modeling:
--------------
The features used for the modeling the different blocks are added to the used_signals directory. Script '' captures the generic flow of power model generation and validation. 
