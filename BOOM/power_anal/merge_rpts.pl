use strict;
use warnings;
#Usage - perl merge_rpts.pl list_of_csv_files_to_merge_in_a_file output_merged_csv_file
#Idea - For each line, parse through that line in all files and append to a separate array and print it to the output file
open(D,">$ARGV[1]");
open(F,"<$ARGV[0]");
my @F = <F>;
close F;
my $file=0;
my $no_of_lines = 0;
open(F2,"<$F[0]");
my @F2 = <F2>;
close F2;
$no_of_lines=@F2;

for(my $i=0;$i<$no_of_lines;$i=$i+1) {
  $file=0;
  my @arr = ();
  for my $j (@F){
    open(F2,"<$j");
    my @F2 = <F2>;
    close F2;
    chomp $F2[$i];
    my @spl_arr = split(',',$F2[$i]);
    if($file==0) {
    $file=$file+1;
  }
  else {
    shift(@spl_arr);
    shift(@spl_arr);
    $file=$file+1;
  }
  for my $k (@spl_arr){
    push(@arr,$k);
  }
}
my $joint = join(',',@arr);
print D  "$joint\n";
}
