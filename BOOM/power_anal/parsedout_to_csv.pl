use strict;use warnings;
open(F,"<$ARGV[0]");
my @F=<F>;
close F;
open(D,">$ARGV[1]");
my $old_power;
my $tset=0;
my $time;
for my $i(@F)
  {
    #printf "Iterating\n";
    chomp $i;
    if($i=~m/e-/)
    {
      #printf "Here - 1\n";
      $old_power = $i;
      
      my @timearr = split(':',$time) if($tset);
      my @powerarr = split(/\s+/,$old_power) if($tset);
      printf D "$timearr[1],$powerarr[1]\n" if($tset);
      #printf "$timearr[1],$powerarr[1]\n" if($tset);
      $tset = 0;
    }elsif($i=~m/e\+/)
    {
      #printf "Here - 2\n";
      $old_power = $i;
      
      my @timearr = split(':',$time) if($tset);
      my @powerarr = split(/\s+/,$old_power) if($tset);
      printf D "$timearr[1],$powerarr[1]\n" if($tset);
      $tset = 0;
    }else
    {
      #printf "Here - 3\n";
      my @timearr = split(':',$time) if($tset);
      my @powerarr = split(/\s+/,$old_power) if($tset);
      printf D "$timearr[1],$powerarr[1]\n" if($tset);
#      if($tset)
    #     {
    #   $tset=0;}
    # else {
      $tset=1;
      #} 
      $time = $i;
    }
  }
