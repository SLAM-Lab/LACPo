source /home/unga/ajaykrishna1111/.synopsys_pt.setup
set power_enable_analysis TRUE
set power_analysis_mode time_based
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
report_timing > timing_with_parasitics.mt-matmul.rpt
set vcd mt-matmul
  reset_switching_activity
  #activity file
  read_vcd -zero_delay  -strip_path "/TestDriver/testHarness/TestHarness/boom_tile" /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero_pd/$vcd.vcd
  report_switching_activity > $vcd.pre_switching_activity.rpt
  #Analyze power
  check_power > $vcd.check_power.rpt
  set_power_analysis_options -waveform_interval 3 -cycle_accurate_clock clock -waveform_format out -waveform_output $vcd
  update_power 
  report_switching_activity > $vcd.switching_activity.rpt
  report_power -nosplit > $vcd.power_summary.rpt
  report_power -nosplit -hier -levels 8 > $vcd.power_hier.rpt


quit
