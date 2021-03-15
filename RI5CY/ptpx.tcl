set power_enable_analysis TRUE
set power_analysis_mode time_based
set netlist <>
set vcd_name <>
set vcd_path <>

read_verilog $netlist
link_design riscv_core
current_design riscv_core
create_clock -name clk_i -period 40 -waveform {0 20} [get_ports clk_i]
reset_switching_activity
set_power_analysis_options -waveform_interval 40 -cycle_accurate_clock clk_i -waveform_format out -waveform_output $vcd_name
read_vcd -zero_delay  -strip_path "tb/top_i/core_region_i/CORE.RISCV_CORE" $vcd_path
report_switching_activity > switching.rpt
update_power
report_power > power_summary.rpt
report_power -hier -levels 4 > power_hier.rpt
