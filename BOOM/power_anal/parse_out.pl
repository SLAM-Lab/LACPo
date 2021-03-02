use strict;
use warnings;
open(F,"<$ARGV[0]");
my @F=<F>;
close F;
open(D,">$ARGV[1]");
my @spl;
my $start_printing=0;
for my $i(@F)
{
    chomp $i;
    if($start_printing==0){
      $start_printing =1 if($i=~m/index/);
    }
    else{

    @spl=split(/\s+/,$i);
    if(defined $spl[1])
    {
      printf D "$i\n";
    }
    else
    {
      printf D "Time: $i\n";
    }
  }

}

