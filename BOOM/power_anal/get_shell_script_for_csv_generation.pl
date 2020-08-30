use strict;
use warnings;
open(F,"<$ARGV[0]");
my @F = <F>;
close F;

open(E,"<$ARGV[1]");
my @E = <E>;
close E;
#Usage
# perl get_shell_script_for_csv_generation.pl <hierarchies_number.fpiu_unit.rpt> <list_of_parsed_out> run_me.fpiu_unit.sh
open(D,">$ARGV[2]");

for my $i (@F){
  chomp $i;
  my @arr = split(',',$i);
  system "mkdir -p $arr[0]";
  for my $j (@E){
    chomp $j;
    #print D "sh run.generic.sh ../$j $j"."."."$arr[0]".".csv $arr[1]\n";
    print D "sh run.generic.sh parsed_csvs/$j $arr[0]"."/"."$j"."."."$arr[0]".".csv $arr[1]\n";
  }
}

