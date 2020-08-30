use strict;
use warnings;

open(F,"<$ARGV[0]");
my @F = <F>;
close F;
open(F2,"<$ARGV[1]");
my @E = <F2>;
close F2;

open(D,">$ARGV[2]");
chomp $F[0];
#my @numbers = split(',',$F[0]);
#my $pr_num = shift(@numbers);
#print "$pr_num\n";
my $iter=1;
print "Hello\n";
#for my $pr_num (@numbers){
for my $pr_num (@F){
  chomp $pr_num;
  print "pr_num : $pr_num\n";
  my @pr_num_arr = split(',',$pr_num);
    $iter=1;
  for my $i (@E){
    #print "iter : $iter\n";
    if($iter==$pr_num_arr[0]){
      my @array = split('\s+',$i);
      #$pr_num = shift(@numbers);
      if($array[0]=~m/BoomTile/){
        print D "$array[0], $pr_num_arr[1], $array[1], $array[2], $array[3], $array[4]\n";
        last;
        }
        else {
          print D "$array[1], $pr_num_arr[1], $array[3], $array[4], $array[5], $array[6]\n";
          last;
        }
      }
    $iter = $iter+1;
    }
  #print "$iter\n";
}




