use strict;
use warnings;
#Usage perl write_stim_file.pl <dump_file> <just_hexa_stim_file>
open(D,"<$ARGV[0]");
my @D = <D>;
close D;
open(D2,">$ARGV[1]");
my $comp_inst_pending;
my $old_comp_inst;
for my $i (@D)
{
  chomp $i;
  my $do = 0;
	if($i=~m/:/)
	{
		if($i=~m/<.*:|file|Disass/)
		{}
		else
		{
			my @arr=split(/\s+/,$i);
      my $line_size = @arr;
#			my @a=split(':',$arr[0]);
#			printf("0x$a[0],$arr[1]:");
      if($arr[-1]=~m/0x/)
      {
	    	my @data = split('',$arr[-1]);
        if($data[0]=~m/0/ && $data[1]=~m/x/)
        {
          shift(@data);
          shift(@data);
          my $data_size = @data;
          if($data_size < 9)
          {
            printf "ignoring"; #comment
            $do = 1;
          }
          else 
          {

           for my $j (1..($line_size-2))
           {
                  
             			my @code = split('',$arr[$j]);
	            		my $size = @code;
	            		printf("$arr[$j]  - $size\n");
	            		if($size==8)
            			{
                  if($comp_inst_pending)
                    {
                      printf D2 "$code[4]"."$code[5]"."$code[6]"."$code[7]"."$old_comp_inst\n";
                      $old_comp_inst ="$code[0]"."$code[1]"."$code[2]"."$code[3]"; 

                    }
                  else 
                    {
          	  		  	printf D2 "$arr[$j]\n"; 
                    }
            			}
            			elsif($size==4)
            			{
                    if ($comp_inst_pending)
                    {
                      printf D2 "$arr[$j]"."$old_comp_inst\n";
                      $comp_inst_pending = 0;
                    }
                    else
                    {
                      $old_comp_inst = $arr[$j];
                      $comp_inst_pending = 1;
                    }
      			      } 
           }
          }
        }
        else
        {
          $do = 1;
        }

      }
      else 
      {
        $do = 1;
      }
      if($do) { 
			my @code = split('',$arr[1]);
			my $size = @code;
			printf("$size\n");
			if($size==8)
			{
        if($comp_inst_pending)
        {
          printf D2 "$code[4]"."$code[5]"."$code[6]"."$code[7]"."$old_comp_inst\n";
          $old_comp_inst ="$code[0]"."$code[1]"."$code[2]"."$code[3]"; 

        }
        else 
        {
				printf D2 "$arr[1]\n"; 
      }
			}
			elsif($size==4)
			{
        if ($comp_inst_pending)
        {
          printf D2 "$arr[1]"."$old_comp_inst\n";
          $comp_inst_pending = 0;
        }
        else
        {
          $old_comp_inst = $arr[1];
          $comp_inst_pending = 1;
        }
       

			}
		}
   }
	}

}
if($comp_inst_pending)
{
  $comp_inst_pending =0;
  printf D2 "0000"."$old_comp_inst\n";
}

