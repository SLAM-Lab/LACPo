use strict;
use warnings;
#usage perl generate_the_signal_extraction_script.pl <block> <core/frontend>
open(D,">$ARGV[0].sh");

printf D "source alias_set.sh\n";
printf D "mkdir -p $ARGV[0]\n";
my @arr = ("dhrystone","median", "multiply", "mm", "mt-matmul", "mt-vvadd", "vvadd", "towers", "spmv", "qsort");
for my $i (@arr){
  if($ARGV[1]=~/core/){
    printf D "vcd2csv ../core/$i.core.vcd /home/local/supreme/ajaykrishna1111/BOOM/Powermodel/nonideal_clock_clk_constraints_nl/power_models/$ARGV[0]/signal_list /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
    printf D "perl add_benchmark_name.pl /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.name_added.csv $i\n";
    printf D "mv  /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.name_added.csv /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
    printf D "sed -i 's///g'  /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
  }
  else { 
    printf D "vcd2csv ../frontend/$i.frontend.vcd /home/local/supreme/ajaykrishna1111/BOOM/Powermodel/nonideal_clock_clk_constraints_nl/power_models/$ARGV[0]/signal_list /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
    printf D "perl add_benchmark_name.pl /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.name_added.csv $i\n";
    printf D "mv  /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.name_added.csv /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
    printf D "sed -i 's///g'  /home/local/supreme/ajaykrishna1111/BOOM/Simulations/vcds_for_medium_init_with_zero/signal_extracted_csv/$ARGV[0]/$i.csv\n";
  }
}

