use strict;
use warnings;
open(F,"<$ARGV[0]");
my @F=<F>;
close F;

open(E,"<$ARGV[1]");
my @E=<E>;
close E;

open(D,">$ARGV[2]");

for my $i (@F) {
  chomp $i;
  print D "$i,";
  for my $j (@E) {
    my @array_new;
    if($j=~m/ $i /) {
      @array_new = split(/\s+/,$j);
      print D "$array_new[0],$array_new[2],";
    }
  }
  print D "\n";
}


