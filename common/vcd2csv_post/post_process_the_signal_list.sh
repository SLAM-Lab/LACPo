sed -i 's/clock$/clock CLOCK/g' signal_list
sed -i 's/\[/ [/g' signal_list 
echo "Found these vectors:"
grep "\[" signal_list  | awk '{print $2}' | sort | uniq
echo "Replacing [16:0]"
sed -i 's/ \[16:0\]/ [16:0] [16:8] [7:0]/g' signal_list
echo "Replacing [19:0]"
sed -i 's/ \[19:0\]/ [19:0] [19:16] [15:8] [7:0]/g' signal_list
echo "Replacing [22:0]"
sed -i 's/ \[22:0\]/ [22:0] [22:16] [15:8] [7:0]/g' signal_list
echo "Replacing [26:0]"
sed -i 's/ \[26:0\]/ [26:0] [26:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [31:0]"
sed -i 's/ \[31:0\]/ [31:0] [31:16] [15:0] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [37:0]"
sed -i 's/ \[37:0\]/ [37:0] [37:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [38:0]"
sed -i 's/ \[38:0\]/ [38:0] [38:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [39:0]"
sed -i 's/ \[39:0\]/ [39:0] [39:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [53:0]"
sed -i 's/ \[53:0\]/ [53:0] [53:48] [47:32] [31:16] [15:0] [53:48] [47:40] [39:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [63:0]"
sed -i 's/ \[63:0\]/ [63:0] [63:32] [31:0] [63:48] [47:32] [31:16] [15:0] [63:56] [55:48] [47:40] [39:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list
echo "Replacing [64:0]"
sed -i 's/ \[64:0\]/ [64:0] [64:32] [31:0] [64:48] [47:32] [31:16] [15:0] [64:56] [55:48] [47:40] [39:32] [31:24] [23:16] [15:8] [7:0]/g' signal_list


