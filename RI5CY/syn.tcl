#################################################################################
# Dongwook Lee, 05/30/2013
# Src: http://classes.soe.ucsc.edu/cmpe125/Spring07/dc_tutorial/dc_tutorial.html
##################################################################################

#set LIB_NAME NangateOpenCellLibrary
set LIB_NAME gscl45nm
set DFF_CELL "DFFSR"
set LIB_DFF_D "$LIB_NAME/$DFF_CELL/D"
set DFF_CKQ 0.11
set DFF_SETUP 0.68

#set search_path  [ list .  /nfs/pdx/disks/stm.disk.01/designkits/nangate/1.3.v2009.07/liberty]
#set link_library [list NangateOpenCellLibrary_typical_conditional_ccs.db dw_foundation.sldb]
#set target_library [list NangateOpenCellLibrary_typical_conditional_ccs.db]
#set_dont_use [get_lib_cells "$LIB_NAME/SDFF*"]
#set_dont_use [get_lib_cells "$LIB_NAME/CLK*"]
#set_dont_use -power [get_lib_cells "$LIB_NAME/CLKGATETST_*"]


# The top module name.
set TOPLEVEL  "riscv_core"
# A subdirectory to put reports in.
set LOG_PATH  "log_new_rtl/"
sh mkdir log_new_rtl
# A subdirectory to put the synthesized (gate-level) verilog output in.
set GATE_PATH  "gate_new_rtl/"
sh mkdir gate_new_rtl
# The subdirectory containing the unsynthesized (behavior) verilog input.
set RTL_PATH  "../rtl/include"
# A name to supplement reports and file names with.
set STAGE  "final"
# The clock input signal name.
set CLK  "clk_i"
# The reset input signal name.
set RST  "rst_ni"
#set SW_RST  
# The target clock period in library units.
set CLK_PERIOD 20.0
# The clock uncertainty in library units. (4.5 * 20%)
set CLK_UNCERTAINTY  4.0

set search_path  [concat $search_path $RTL_PATH $GATE_PATH]
# read the verilog in a "bottom up" manner to reduce the numbe of unresolved warning messages
source synth_filelist
#define_design_lib work -path ./work
#
#analyze -format verilog -work work ../verilog/dct.v
read_file $l -autoread -top $TOPLEVEL
elaborate $TOPLEVEL
current_design $TOPLEVEL

link
#uniquify
# check the design for errors such as missing module definisions
check_design > $LOG_PATH$TOPLEVEL-check_design2.log

create_clock $CLK -period $CLK_PERIOD
set_clock_uncertainty $CLK_UNCERTAINTY [all_clocks]
set_dont_touch_network [all_clocks]

remove_driving_cell $RST
set_drive 0 $RST
set_dont_touch_network $RST
#remove_driving_cell $SW_RST
#set_drive 0 $SW_RST
#set_dont_touch_network $SW_RST

set_output_delay $DFF_SETUP -clock $CLK [all_outputs]
set_load [expr [load_of $LIB_DFF_D] * 4] [all_outputs]

set all_inputs_wo_rst_clk [remove_from_collection [remove_from_collection [all_inputs] [get_port $CLK]] [get_port $RST]] 
set_input_delay -clock $CLK $DFF_CKQ $all_inputs_wo_rst_clk
set_driving_cell -library $LIB_NAME -lib_cell $DFF_CELL -pin Q $all_inputs_wo_rst_clk
#ungroup -flatten -all 
compile_ultra -gate_clock -no_uniquify -no_autoungroup -no_boundary_optimization
check_design > $LOG_PATH$TOPLEVEL-check_design.log
#set_fix_hold $CLK
#compile -only_design_rule -incremental_mapping
#compile_ultra
set_fix_multiple_port_nets -all -constants -buffer_constants [get_designs *]
remove_unconnected_ports -blast_buses [get_cells -hierarchical *]

set bus_inference_style {%s[%d]}
set bus_naming_style {%s[%d]}
set hdlout_internal_busses true
#change_names -hierarchy -rule verilog
#define_name_rules name_rule -allowed "A-Z a-z 0-9 _" -max_length 255 -type cell
#define_name_rules name_rule -allowed "A-Z a-z 0-9 _[]" -max_length 255 -type net
#define_name_rules name_rule -map {{"\\*cell\\*" "cell" }} 
#define_name_rules name_rule -case_insensitive
#change_names -hierarchy -rules name_rule
write -hierarchy -format ddc  -output $GATE_PATH/$TOPLEVEL-$STAGE.ddc
write_sdc   $GATE_PATH/$TOPLEVEL-$STAGE.sdc
write_sdf   $GATE_PATH/$TOPLEVEL-$STAGE.sdf
write -hierarchy -format verilog -output $GATE_PATH/$TOPLEVEL-$STAGE.v

report_area     > $LOG_PATH/$TOPLEVEL-$STAGE-area.log                
report_timing -nworst 10        > $LOG_PATH/$TOPLEVEL-$STAGE-timing.log
report_hierarchy                > $LOG_PATH/$TOPLEVEL-$STAGE-hierarchy.log
report_resources                > $LOG_PATH/$TOPLEVEL-$STAGE-resources.log
#report_references               > $LOG_PATH/$TOPLEVEL-$STAGE-references.log
report_constraint               > $LOG_PATH/$TOPLEVEL-$STAGE-constraint.log
#report_ultra_optimizations      > $LOG_PATH/$TOPLEVEL-$STAGE-ultra_optimizations.log
quit
