use strict;
use warnings;
open(F,"<$ARGV[0]");
my @F = <F>;
close F;

open(D,">$ARGV[1]");
my $iter = 0;
for my $i (@F){
  chomp $i;
  if($iter==0){
    printf D "$i".",Benchmark\n";
    $iter = 1;
  }
  else {
    printf D "$i".",$ARGV[2]\n";
  }
}
close D;
