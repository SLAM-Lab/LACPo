setenv RTL_PATH pulpino-master/rtl
setenv TB_PATH pulpino-master/tb
vhdlan -full64 -debug -f vhdl_master -l compile.log
vlogan -full64 -debug -sverilog -f master_gate -ignore all -timescale=1ns/1ns -l compile.log
vcs -full64 tb -l c.log -debug_all -R +vcs+vcdplusmemon +vcs+dumpvars+<output>.vcd
