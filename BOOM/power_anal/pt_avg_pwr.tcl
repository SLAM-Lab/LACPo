source /home/unga/ajaykrishna1111/.synopsys_pt.setup
set power_enable_analysis TRUE
set power_analysis_mode averaged
read_verilog ../synth_PD/BoomTile.v
link_design BoomTile
current_instance
#Parasitics
read_sdc ../synth_PD/BoomTile.mapped.clk_changed.sdc
set timing_all_clocks_propagated true
read_parasitics ../synth_PD/BoomTile.spef 
set_propagated_clock [get_clock]
check_timing
update_timing
report_timing > timing_with_parasitics.avg.rpt
source vcd_path
foreach vcd $vcd_path {
  reset_switching_activity
  #activity file
  read_vcd -zero_delay  -strip_path "/TestDriver/testHarness/TestHarness/boom_tile" ../simulations/$vcd.vcd
  #Analyze power
  check_power > $vcd.check_power.avg.rpt
  #set_power_analysis_options -waveform_interval 3 -cycle_accurate_clock clock -waveform_format out -waveform_output $vcd
  update_power 
  report_switching_activity > $vcd.switching_activity.avg.rpt
  report_power -nosplit > $vcd.power_summary.avg.rpt
  report_power -nosplit -hier -levels 8 > $vcd.power_hier.avg.rpt
}


quit
